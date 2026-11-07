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
"""Fused 92-layer TP32xDCP8 NVFP4 Pallas megakernel for Qwen3.8-2.4T."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, Unpack, cast

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.collectives import kernel_axis_index

from . import compute, dma, plan

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from jax._src.state.types import TransformedRef

    from .shapes import Qwen38Config

PAGES_PER_GDN_SLOT: int = 5
_RANK_POOL_NDIM: int = 6
_UNRANKED_POOL_NDIM: int = 5
_POOL_7D_NDIM: int = 7
_NORM_WQKV_NDIM: int = 5


class MegakernelDeviceError(RuntimeError):
    """Raised when the JAX device mesh cannot satisfy megakernel rank count."""


@dataclass(frozen=True)
class AttentionRefs:
    """References and scratch buffers for attention subblock."""

    page_table_ref: jax.Array
    pool_out_ref: jax.Array
    w_refs: dict[str, jax.Array]
    scratch: dict[str, jax.Array]
    wo_buf: jax.Array
    me: jax.Array
    axis_name: str | None = "tp"


@dataclass(frozen=True)
class GdnRefs:
    """References and scratch buffers for GDN recurrent subblock."""

    conv_out_ref: jax.Array
    rec_out_ref: jax.Array
    w_refs: dict[str, jax.Array]
    scratch: dict[str, jax.Array]
    wout_buf: jax.Array


@dataclass(frozen=True)
class MoeRefs:
    """References and scratch buffers for MoE subblock."""

    forced_layer: jax.Array | None
    w_refs: dict[str, jax.Array]
    scratch: dict[str, jax.Array]
    wse2_buf: jax.Array


class PallasStepOptions(TypedDict, total=False):
    """Optional settings for make_pallas_step_fn."""

    has_forced_routing: bool
    interpret: bool | pltpu.InterpretParams
    axis_name: str | None


class ShardedStepOptions(TypedDict, total=False):
    """Optional settings for build_sharded_step_fn."""

    has_forced_routing: bool
    interpret: bool | pltpu.InterpretParams
    axis_name: str


class StepOptions(TypedDict, total=False):
    """Optional execution flags for run_qwen38_dcp8_step."""

    use_forced_routing: bool
    return_kv_updates: bool
    interpret: bool | pltpu.InterpretParams | None
    mesh: Mesh | None


def _h_tiled_fp4_matmul(
    x_bf16: jax.Array,
    w_hbm_pair: tuple[jax.Array, jax.Array],
    layer_idx: int | jax.Array,
    vmem_bufs: tuple[jax.Array, jax.Array, tuple[object, object], jax.Array],
    cfg: Qwen38Config,
) -> jax.Array:
    """Stream `(n_htiles, h_tile, N_half)` FP4 weights over `h_tile`."""
    w_hbm_4d, s_hbm_4d = w_hbm_pair
    w_buf_pingpong, s_buf_pingpong, sem_pair, fp4_out_buf = vmem_bufs
    n_ht = cfg.H // cfg.h_tile
    cp_w0 = dma.copy_4d_htile(w_hbm_4d, layer_idx, 0, w_buf_pingpong.at[0], sem_pair[0])
    cp_s0 = dma.copy_4d_htile(s_hbm_4d, layer_idx, 0, s_buf_pingpong.at[0], sem_pair[1])
    cp_w0.wait()
    cp_s0.wait()

    if n_ht == 1:
        return compute.fp4_matmul(
            x_bf16,
            w_buf_pingpong[0, :, :],
            s_buf_pingpong[0, :, :],
            group_size=cfg.fp4_group,
        )

    out_cols = w_buf_pingpong.shape[2] * 2
    fp4_out_buf[:, :out_cols] = jnp.zeros((cfg.B, out_cols), dtype=jnp.float32)

    def _tile_body(ht: jax.Array, _: None) -> None:
        """Process one horizontal tile step and issue ping-pong copies."""
        slot = ht & 1
        nxt_slot = 1 - slot
        has_next = (ht + 1) < n_ht

        @pl.when(has_next)
        def _prefetch() -> None:
            """Prefetch next weight and scale tile into inactive buffer slot."""
            dma.copy_4d_htile(
                w_hbm_4d, layer_idx, ht + 1, w_buf_pingpong.at[nxt_slot], sem_pair[0]
            )
            dma.copy_4d_htile(
                s_hbm_4d, layer_idx, ht + 1, s_buf_pingpong.at[nxt_slot], sem_pair[1]
            )

        x_slice = jax.lax.dynamic_slice_in_dim(
            x_bf16, ht * cfg.h_tile, cfg.h_tile, axis=1
        )
        part = compute.fp4_matmul(
            x_slice,
            w_buf_pingpong[slot, :, :],
            s_buf_pingpong[slot, :, :],
            group_size=cfg.fp4_group,
        )
        fp4_out_buf[:, :out_cols] = fp4_out_buf[:, :out_cols] + part

        @pl.when(has_next)
        def _wait_next() -> None:
            """Wait for completion of prefetched weight and scale tiles."""
            d2_w, d3_w = w_hbm_4d.shape[2], w_hbm_4d.shape[3]
            d2_s, d3_s = s_hbm_4d.shape[2], s_hbm_4d.shape[3]
            pltpu.make_async_copy(
                w_hbm_4d.at[layer_idx, ht + 1, pl.ds(0, d2_w), pl.ds(0, d3_w)],
                w_buf_pingpong.at[nxt_slot],
                sem_pair[0],
            ).wait()
            pltpu.make_async_copy(
                s_hbm_4d.at[layer_idx, ht + 1, pl.ds(0, d2_s), pl.ds(0, d3_s)],
                s_buf_pingpong.at[nxt_slot],
                sem_pair[1],
            ).wait()

    jax.lax.fori_loop(0, n_ht, _tile_body, None)
    return fp4_out_buf[:, :out_cols]


def _h_tiled_expert_fp4_matmul(
    x_bf16: jax.Array,
    w_hbm_pair: tuple[jax.Array, jax.Array],
    coords: tuple[int | jax.Array, int | jax.Array],
    vmem_bufs: tuple[jax.Array, jax.Array, tuple[object, object], jax.Array],
    cfg: Qwen38Config,
) -> jax.Array:
    """Stream `(n_htiles, h_tile, N_half)` FP4 expert weights over `h_tile`."""
    we13_hbm_5d, we13_s_hbm_5d = w_hbm_pair
    layer_idx, expert_idx = coords
    we13_buf, we13_s_buf, sem_pair, fp4_out_buf = vmem_bufs
    n_ht = cfg.H // cfg.h_tile
    cp_w0 = dma.copy_5d_expert_htile(
        we13_hbm_5d, (layer_idx, expert_idx, 0), we13_buf.at[0], sem_pair[0]
    )
    cp_s0 = dma.copy_5d_expert_htile(
        we13_s_hbm_5d, (layer_idx, expert_idx, 0), we13_s_buf.at[0], sem_pair[1]
    )
    cp_w0.wait()
    cp_s0.wait()

    if n_ht == 1:
        return compute.fp4_matmul(
            x_bf16,
            we13_buf[0, :, :],
            we13_s_buf[0, :, :],
            group_size=cfg.fp4_group,
        )

    out_cols = we13_buf.shape[2] * 2
    rows = x_bf16.shape[0]
    fp4_out_buf[:rows, :out_cols] = jnp.zeros((rows, out_cols), dtype=jnp.float32)

    def _tile_body(ht: jax.Array, _: None) -> None:
        """Process one horizontal tile step for the selected expert."""
        slot = ht & 1
        nxt_slot = 1 - slot
        has_next = (ht + 1) < n_ht

        @pl.when(has_next)
        def _prefetch() -> None:
            """Prefetch next expert weight and scale tile into inactive slot."""
            dma.copy_5d_expert_htile(
                we13_hbm_5d,
                (layer_idx, expert_idx, ht + 1),
                we13_buf.at[nxt_slot],
                sem_pair[0],
            )
            dma.copy_5d_expert_htile(
                we13_s_hbm_5d,
                (layer_idx, expert_idx, ht + 1),
                we13_s_buf.at[nxt_slot],
                sem_pair[1],
            )

        x_slice = jax.lax.dynamic_slice_in_dim(
            x_bf16, ht * cfg.h_tile, cfg.h_tile, axis=1
        )
        part = compute.fp4_matmul(
            x_slice,
            we13_buf[slot, :, :],
            we13_s_buf[slot, :, :],
            group_size=cfg.fp4_group,
        )
        fp4_out_buf[:rows, :out_cols] = fp4_out_buf[:rows, :out_cols] + part

        @pl.when(has_next)
        def _wait_next() -> None:
            """Wait for completion of prefetched expert weight and scale tiles."""
            d3_w, d4_w = we13_hbm_5d.shape[3], we13_hbm_5d.shape[4]
            d3_s, d4_s = we13_s_hbm_5d.shape[3], we13_s_hbm_5d.shape[4]
            pltpu.make_async_copy(
                we13_hbm_5d.at[
                    layer_idx, expert_idx, ht + 1, pl.ds(0, d3_w), pl.ds(0, d4_w)
                ],
                we13_buf.at[nxt_slot],
                sem_pair[0],
            ).wait()
            pltpu.make_async_copy(
                we13_s_hbm_5d.at[
                    layer_idx, expert_idx, ht + 1, pl.ds(0, d3_s), pl.ds(0, d4_s)
                ],
                we13_s_buf.at[nxt_slot],
                sem_pair[1],
            ).wait()

    jax.lax.fori_loop(0, n_ht, _tile_body, None)
    return fp4_out_buf[:rows, :out_cols]


def _write_kv_for_batch(
    att_idx: int,
    pos_vec: jax.Array,
    page_pool_refs: tuple[jax.Array, jax.Array, dict[str, jax.Array]],
    kv_state: tuple[jax.Array, jax.Array, jax.Array],
    cfg: Qwen38Config,
) -> None:
    """Execute KV cache write scattering across DCP ranks for current batch."""
    page_table_ref, pool_out_ref, scratch = page_pool_refs
    k_fp8, v_fp8, dcp_slot = kv_state
    kv_write_buf = scratch["kv_write_buf"]
    kv_sem = scratch["kv_sem"]
    d_dim = cfg.D
    k_r = cfg.K_R
    page_sz = cfg.page_size

    for b in range(cfg.B):
        pos_b = pos_vec[b]
        valid_req = pos_b >= 0
        safe_pos = jnp.maximum(pos_b, 0)
        global_page = safe_pos // page_sz
        owner_dcp = global_page % cfg.DCP
        raw_local_page = global_page // cfg.DCP
        in_page_bounds = raw_local_page < cfg.max_pages_per_seq
        local_page = jnp.minimum(raw_local_page, cfg.max_pages_per_seq - 1)
        offset_in_page = safe_pos % page_sz
        phys_page = jnp.maximum(page_table_ref[att_idx, b, local_page], 0)

        @pl.when(valid_req & in_page_bounds & (owner_dcp == dcp_slot))
        def _write_kv(
            phys_page: jax.Array = phys_page,
            offset_in_page: jax.Array = offset_in_page,
            b_idx: int = b,
        ) -> None:
            """Write key and value states into physical page slice."""
            cp_in = pltpu.make_async_copy(
                pool_out_ref.at[
                    att_idx,
                    phys_page,
                    pl.ds(0, 2),
                    pl.ds(0, k_r),
                    pl.ds(0, d_dim),
                    pl.ds(0, page_sz),
                ],
                kv_write_buf,
                kv_sem.at[2],
            )
            cp_in.start()
            cp_in.wait()

            tok_idx = jnp.arange(page_sz, dtype=jnp.int32)[None, None, :]
            is_target = tok_idx == offset_in_page
            k_b = jnp.broadcast_to(k_fp8[b_idx, :, :, None], (k_r, d_dim, page_sz))
            v_b = jnp.broadcast_to(v_fp8[b_idx, :, :, None], (k_r, d_dim, page_sz))
            kv_write_buf[0, :, :, :] = jnp.where(
                is_target, k_b, kv_write_buf[0, :, :, :]
            )
            kv_write_buf[1, :, :, :] = jnp.where(
                is_target, v_b, kv_write_buf[1, :, :, :]
            )

            cp_out = pltpu.make_async_copy(
                kv_write_buf,
                pool_out_ref.at[
                    att_idx,
                    phys_page,
                    pl.ds(0, 2),
                    pl.ds(0, k_r),
                    pl.ds(0, d_dim),
                    pl.ds(0, page_sz),
                ],
                kv_sem.at[3],
            )
            cp_out.start()
            cp_out.wait()


def _compute_dcp_attention_scores(
    att_idx: int,
    pos_vec: jax.Array,
    q_dcp_all: jax.Array,
    refs: AttentionRefs,
    cfg: Qwen38Config,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Execute local online softmax scoring across DCP pages for all batch items."""
    b_dim = cfg.B
    d_dim = cfg.D
    dcp = cfg.DCP
    dcp_slot = refs.me % dcp
    max_pages = cfg.max_pages_per_seq
    kv_page_buf = refs.scratch["kv_page_buf"]
    kv_sem = refs.scratch["kv_sem"]
    dcp_q = cfg.DCP * cfg.Q_R

    m_list, l_list, acc_list = [], [], []
    for b in range(b_dim):
        pos_b = pos_vec[b]
        n_pages_b = jnp.clip(
            (jnp.maximum(pos_b, 0) // (cfg.page_size * dcp)) + 1,
            1,
            max_pages,
        )
        q_b = q_dcp_all[b]
        m_b = jnp.full((dcp_q,), -1e30, dtype=jnp.float32)
        l_b = jnp.zeros((dcp_q,), dtype=jnp.float32)
        acc_b = jnp.zeros((dcp_q, d_dim), dtype=jnp.float32)

        first_phys = jnp.maximum(refs.page_table_ref[att_idx, b, 0], 0)
        cp_p0 = dma.copy_kv_page_in(
            refs.pool_out_ref, att_idx, first_phys, kv_page_buf.at[0], kv_sem.at[0]
        )
        cp_p0.wait()

        def _page_body(
            lp: jax.Array,
            carry: tuple[jax.Array, jax.Array, jax.Array],
            page_ctx: tuple[int, jax.Array, jax.Array, jax.Array] = (
                b,
                n_pages_b,
                pos_b,
                q_b,
            ),
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Update online softmax state for one local DCP page."""
            b_idx, n_pages, pos_val, q_val = page_ctx
            m_cur, l_cur, acc_cur = carry
            slot = lp & 1
            nxt_slot = 1 - slot
            has_next = (lp + 1) < n_pages
            nxt_lp = jnp.minimum(lp + 1, max_pages - 1)
            nxt_phys = jnp.maximum(refs.page_table_ref[att_idx, b_idx, nxt_lp], 0)

            @pl.when(has_next)
            def _prefetch_page() -> None:
                """Issue asynchronous DMA prefetch for the next page tile."""
                dma.copy_kv_page_in(
                    refs.pool_out_ref,
                    att_idx,
                    nxt_phys,
                    kv_page_buf.at[nxt_slot],
                    kv_sem.at[nxt_slot],
                )

            k_pg = kv_page_buf[slot, 0, :, :, :]
            v_pg = kv_page_buf[slot, 1, :, :, :]
            tok_pos = (lp * dcp + dcp_slot) * cfg.page_size + jnp.arange(
                cfg.page_size, dtype=jnp.int32
            )
            m_next, l_next, acc_next = compute.dcp_online_softmax_update(
                q_val, (k_pg, v_pg), (pos_val, tok_pos), (m_cur, l_cur, acc_cur), cfg
            )

            @pl.when(has_next)
            def _wait_next_page() -> None:
                """Wait on DMA completion for the prefetched page tile."""
                _, _, d1, d2, d3, d4 = refs.pool_out_ref.shape
                pltpu.make_async_copy(
                    refs.pool_out_ref.at[
                        att_idx,
                        nxt_phys,
                        pl.ds(0, d1),
                        pl.ds(0, d2),
                        pl.ds(0, d3),
                        pl.ds(0, d4),
                    ],
                    kv_page_buf.at[nxt_slot],
                    kv_sem.at[nxt_slot],
                ).wait()

            return m_next, l_next, acc_next

        m_b, l_b, acc_b = jax.lax.fori_loop(0, n_pages_b, _page_body, (m_b, l_b, acc_b))
        m_list.append(m_b)
        l_list.append(l_b)
        acc_list.append(acc_b)

    m_local = jnp.stack(m_list, axis=0)
    l_local = jnp.stack(l_list, axis=0)
    acc_local = jnp.stack(acc_list, axis=0)
    return m_local, l_local, acc_local


def _run_attention_subblock(
    att_idx: int,
    x_norm: jax.Array,
    pos_vec: jax.Array,
    refs: AttentionRefs,
    cfg: Qwen38Config,
) -> jax.Array:
    """Execute one full-attention layer with DCP=8 KV cache update and LSE merge."""
    b_dim = cfg.B
    d_dim = cfg.D
    dcp = cfg.DCP
    q_r = cfg.Q_R
    k_r = cfg.K_R
    w_sem = refs.scratch["w_sem"]

    cp_wo = dma.copy_3d_tile(refs.w_refs["wo"], att_idx, refs.wo_buf, w_sem.at[2])
    cp_wos = dma.copy_3d_tile(
        refs.w_refs["wo_s"],
        att_idx,
        refs.scratch["wo_s_buf"].at[pl.ds(0, refs.w_refs["wo_s"].shape[1]), :],
        w_sem.at[3],
    )

    hbm_pair = (refs.w_refs["wqkv"], refs.w_refs["wqkv_s"])
    vmem_bufs = (
        refs.scratch["wqkv_buf"],
        refs.scratch["wqkv_s_buf"],
        (w_sem.at[0], w_sem.at[1]),
        refs.scratch["fp4_out_buf"],
    )
    qkv_f32 = _h_tiled_fp4_matmul(x_norm, hbm_pair, att_idx, vmem_bufs, cfg)

    qg_dim = 2 * q_r * d_dim
    k_dim = k_r * d_dim
    qg_all = qkv_f32[:, :qg_dim].reshape(b_dim, q_r, 2 * d_dim)
    q_raw = qg_all[:, :, :d_dim].astype(jnp.bfloat16)
    q_gate = qg_all[:, :, d_dim:].astype(jnp.bfloat16)
    k_raw = (
        qkv_f32[:, qg_dim : qg_dim + k_dim]
        .reshape(b_dim, k_r, d_dim)
        .astype(jnp.bfloat16)
    )
    v_raw = qkv_f32[:, qg_dim + k_dim :].reshape(b_dim, k_r, d_dim).astype(jnp.bfloat16)

    q_norm_w = refs.w_refs["q_norm"][att_idx, :][None, None, :]
    k_norm_w = refs.w_refs["k_norm"][att_idx, :][None, None, :]
    q_normed = compute.rmsnorm(q_raw, q_norm_w)
    k_normed = compute.rmsnorm(k_raw, k_norm_w)

    q_rot = compute.partial_rope(q_normed, pos_vec, cfg.rope_dim)
    k_rot = compute.partial_rope(k_normed, pos_vec, cfg.rope_dim)

    q_buffers = dma.DcpQueryBuffers(
        q_local_send=refs.scratch["q_local_send"],
        q_dcp_recv=refs.scratch["q_dcp_recv"],
        dcp_q_ssem=refs.scratch["dcp_q_ssem"],
        dcp_q_rsem=refs.scratch["dcp_q_rsem"],
    )
    q_copies = dma.dcp_group_query_start(
        q_buffers,
        q_rot,
        refs.me,
        cfg,
        axis_name=refs.axis_name,
    )

    k_fp8 = k_rot.astype(jnp.float8_e4m3fn)
    v_fp8 = v_raw.astype(jnp.float8_e4m3fn)
    dcp_slot = refs.me % dcp

    _write_kv_for_batch(
        att_idx,
        pos_vec,
        (refs.page_table_ref, refs.pool_out_ref, refs.scratch),
        (k_fp8, v_fp8, dcp_slot),
        cfg,
    )

    dma.dcp_group_query_wait(q_copies)

    q_dcp_all = jnp.transpose(refs.scratch["q_dcp_recv"][...], (1, 0, 2, 3)).reshape(
        b_dim, dcp * q_r, d_dim
    )

    m_local, l_local, acc_local = _compute_dcp_attention_scores(
        att_idx, pos_vec, q_dcp_all, refs, cfg
    )

    lse_buffers = dma.DcpLseBuffers(
        lse_part_send=refs.scratch["lse_part_send"],
        lse_part_recv=refs.scratch["lse_part_recv"],
        dcp_lse_ssem=refs.scratch["dcp_lse_ssem"],
        dcp_lse_rsem=refs.scratch["dcp_lse_rsem"],
    )
    dma.dcp_lse_exchange(
        lse_buffers,
        (m_local, l_local, acc_local),
        refs.me,
        cfg,
        axis_name=refs.axis_name,
    )
    acc_flat = compute.dcp_merge_lse(
        refs.scratch["lse_part_recv"][...],
        q_gate,
        dcp_slot,
        cfg,
    )

    cp_wo.wait()
    cp_wos.wait()

    w_vmem = refs.wo_buf[:, :]
    s_vmem = refs.scratch["wo_s_buf"][: refs.w_refs["wo_s"].shape[1], :]
    out_proj = compute.fp4_matmul(acc_flat, w_vmem, s_vmem, group_size=cfg.fp4_group)
    return out_proj.astype(jnp.bfloat16)


def _run_gdn_subblock(
    gdn_idx: int,
    x_norm: jax.Array,
    refs: GdnRefs,
    cfg: Qwen38Config,
) -> jax.Array:
    """Execute one GDN layer with conv1d + recurrent step and output projection."""
    w_sem = refs.scratch["w_sem"]
    state_sem = refs.scratch["state_sem"]
    state_slot = gdn_idx & 1
    conv_slot = cast("dma.BufferSlotRef", refs.scratch["conv_buf"].at[state_slot])
    rec_slot = cast("dma.BufferSlotRef", refs.scratch["rec_buf"].at[state_slot])

    cp_cin_c, cp_cin_r = dma.copy_gdn_states_in(
        (refs.conv_out_ref, refs.rec_out_ref),
        gdn_idx,
        (conv_slot, rec_slot),
        (state_sem.at[0], state_sem.at[1]),
    )
    cp_wba = dma.copy_3d_tile(
        refs.w_refs["w_ba"], gdn_idx, refs.scratch["wba_buf"], w_sem.at[2]
    )
    cp_wc = dma.copy_3d_tile(
        refs.w_refs["w_conv"], gdn_idx, refs.scratch["wconv_buf"], w_sem.at[3]
    )

    hbm_pair = (refs.w_refs["w_inproj"], refs.w_refs["w_inproj_s"])
    vmem_bufs = (
        refs.scratch["win_buf"],
        refs.scratch["win_s_buf"],
        (w_sem.at[0], w_sem.at[1]),
        refs.scratch["fp4_out_buf"],
    )
    in_proj = _h_tiled_fp4_matmul(x_norm, hbm_pair, gdn_idx, vmem_bufs, cfg)

    cp_wba.wait()
    cp_wc.wait()
    cp_cin_c.wait()
    cp_cin_r.wait()

    cp_wo = dma.copy_3d_tile(refs.w_refs["w_out"], gdn_idx, refs.wout_buf, w_sem.at[0])
    cp_wos = dma.copy_3d_tile(
        refs.w_refs["w_out_s"],
        gdn_idx,
        refs.scratch["wout_s_buf"].at[pl.ds(0, refs.w_refs["w_out_s"].shape[1]), :],
        w_sem.at[1],
    )

    ba = jnp.dot(
        x_norm, refs.scratch["wba_buf"][:, :], preferred_element_type=jnp.float32
    )
    qkv_part = in_proj[:, : cfg.QKV].astype(jnp.bfloat16)
    z_part = in_proj[:, cfg.QKV : cfg.QKV + cfg.Z_R].astype(jnp.bfloat16)

    conv_w = refs.scratch["wconv_buf"][...]
    a_log = refs.w_refs["a_log"][gdn_idx, :]
    dt_bias = refs.w_refs["dt_bias"][gdn_idx, :]
    gdn_norm = refs.w_refs["gdn_norm"][gdn_idx, :][None, :]

    y_gated, conv_state_out, rec_state_out = compute.gdn_recurrent_step(
        (qkv_part, z_part, ba),
        (conv_slot[...], rec_slot[...]),
        (conv_w, a_log, dt_bias, gdn_norm),
        cfg,
    )

    conv_slot[...] = conv_state_out
    rec_slot[...] = rec_state_out
    cp_cout_c, cp_cout_r = dma.copy_gdn_states_out(
        (conv_slot, rec_slot),
        (refs.conv_out_ref, refs.rec_out_ref),
        gdn_idx,
        (state_sem.at[2], state_sem.at[3]),
    )

    cp_wo.wait()
    cp_wos.wait()

    w_vmem = refs.wout_buf[:, :]
    s_vmem = refs.scratch["wout_s_buf"][: refs.w_refs["w_out_s"].shape[1], :]
    out_proj = compute.fp4_matmul(y_gated, w_vmem, s_vmem, group_size=cfg.fp4_group)
    cp_cout_c.wait()
    cp_cout_r.wait()
    return out_proj.astype(jnp.bfloat16)


def _run_moe_subblock(
    layer_idx: int,
    x_norm: jax.Array,
    refs: MoeRefs,
    cfg: Qwen38Config,
) -> jax.Array:
    """Execute MoE routing, top-K expert compute, shared expert, and all-reduce."""
    b_dim = cfg.B
    k_exp = cfg.K
    w_sem = refs.scratch["w_sem"]

    cp_wg = dma.copy_3d_tile(
        refs.w_refs["w_gate"], layer_idx, refs.scratch["wgate_buf"], w_sem.at[0]
    )
    cp_wseg = dma.copy_3d_tile(
        refs.w_refs["w_se_gate"], layer_idx, refs.scratch["wsegate_buf"], w_sem.at[1]
    )
    cp_wse2 = dma.copy_3d_tile(
        refs.w_refs["w_se2"], layer_idx, refs.wse2_buf, w_sem.at[2]
    )
    cp_wse2_s = dma.copy_3d_tile(
        refs.w_refs["w_se2_s"],
        layer_idx,
        refs.scratch["wse2_s_buf"].at[pl.ds(0, refs.w_refs["w_se2_s"].shape[1]), :],
        w_sem.at[3],
    )
    cp_wg.wait()
    cp_wseg.wait()

    topk_ids, topk_weights = compute.moe_router_topk(
        x_norm, refs.scratch["wgate_buf"][:, :], cfg, forced_ids=refs.forced_layer
    )
    se_gate = jax.nn.sigmoid(
        jnp.dot(
            x_norm,
            refs.scratch["wsegate_buf"][:, :],
            preferred_element_type=jnp.float32,
        )
    )

    hbm_pair_se = (refs.w_refs["w_se13"], refs.w_refs["w_se13_s"])
    vmem_bufs_se = (
        refs.scratch["wse13_buf"],
        refs.scratch["wse13_s_buf"],
        (w_sem.at[0], w_sem.at[1]),
        refs.scratch["fp4_out_buf"],
    )
    se13_f32 = _h_tiled_fp4_matmul(x_norm, hbm_pair_se, layer_idx, vmem_bufs_se, cfg)
    inter_se = compute.swiglu_split(se13_f32, cfg.SE_R)

    cp_wse2.wait()
    cp_wse2_s.wait()
    wse2_vmem = refs.wse2_buf[:, :]
    wse2_s_vmem = refs.scratch["wse2_s_buf"][: refs.w_refs["w_se2_s"].shape[1], :]
    se_down = compute.fp4_matmul(
        inter_se, wse2_vmem, wse2_s_vmem, group_size=cfg.fp4_group
    )
    moe_acc = se_down * se_gate

    e13_hbm_pair = (refs.w_refs["w_e13"], refs.w_refs["w_e13_s"])
    e13_vmem_bufs = (
        refs.scratch["we13_buf"],
        refs.scratch["we13_s_buf"],
        (w_sem.at[0], w_sem.at[1]),
        refs.scratch["fp4_out_buf"],
    )

    for b in range(b_dim):
        x_row = x_norm[b : b + 1, :]
        row_acc = jnp.zeros((1, cfg.H), dtype=jnp.float32)
        for k_idx in range(k_exp):
            exp_id = topk_ids[b, k_idx]
            weight_b = topk_weights[b, k_idx]

            cp_we2 = dma.copy_4d_expert_full(
                refs.w_refs["w_e2"],
                layer_idx,
                exp_id,
                refs.scratch["we2_buf"],
                w_sem.at[2],
            )
            s_slice = pl.ds(0, refs.w_refs["w_e2_s"].shape[2])
            cp_we2_s = dma.copy_4d_expert_full(
                refs.w_refs["w_e2_s"],
                layer_idx,
                exp_id,
                refs.scratch["we2_s_buf"].at[s_slice, :],
                w_sem.at[3],
            )

            e13_row = _h_tiled_expert_fp4_matmul(
                x_row, e13_hbm_pair, (layer_idx, exp_id), e13_vmem_bufs, cfg
            )
            act_b = compute.swiglu_split(e13_row, cfg.I_R)

            cp_we2.wait()
            cp_we2_s.wait()

            we2_vmem = refs.scratch["we2_buf"][:, :]
            we2_s_vmem = refs.scratch["we2_s_buf"][: refs.w_refs["w_e2_s"].shape[2], :]
            down_row = compute.fp4_matmul(
                act_b, we2_vmem, we2_s_vmem, group_size=cfg.fp4_group
            )
            row_acc = row_acc + down_row * weight_b
        row_mask = (jnp.arange(b_dim, dtype=jnp.int32) == b).astype(jnp.float32)[
            :, None
        ]
        moe_acc = moe_acc + jnp.broadcast_to(row_acc, (b_dim, cfg.H)) * row_mask

    return moe_acc.astype(jnp.bfloat16)


decode_att_sublayer = _run_attention_subblock
decode_gdn_sublayer = _run_gdn_subblock
decode_moe_sublayer = _run_moe_subblock


def _unpack_kernel_args(
    args: tuple[jax.Array, ...],
    *,
    has_forced_routing: bool,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
    jax.Array | None,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    dict[str, jax.Array],
]:
    """Unpack positional arguments passed into the megakernel grid."""
    if has_forced_routing:
        (
            x_in_ref,
            pos_ref,
            page_table_ref,
            _,
            _,
            _,
            w_refs,
            forced_ref,
            x_out_ref,
            pool_out_ref,
            conv_out_ref,
            rec_out_ref,
            scratch,
        ) = args
    else:
        (
            x_in_ref,
            pos_ref,
            page_table_ref,
            _,
            _,
            _,
            w_refs,
            x_out_ref,
            pool_out_ref,
            conv_out_ref,
            rec_out_ref,
            scratch,
        ) = args
        forced_ref = None
    return (
        x_in_ref,
        pos_ref,
        page_table_ref,
        cast("dict[str, jax.Array]", w_refs),
        forced_ref,
        x_out_ref,
        pool_out_ref,
        conv_out_ref,
        rec_out_ref,
        cast("dict[str, jax.Array]", scratch),
    )


def build_qwen38_kernel_fn(
    cfg: Qwen38Config,
    *,
    has_forced_routing: bool = False,
    axis_name: str | None = "tp",
) -> Callable[..., None]:
    """Compile the single-step megakernel into a callable Pallas body function."""
    stage_plan, stage_slices = plan.build_stage_scratch_plan(cfg)
    wo_slice = stage_slices["wo_buf"]
    wout_slice = stage_slices["wout_buf"]
    wse2_slice = stage_slices["wse2_buf"]

    def _kernel(*args: jax.Array) -> None:
        """Execute the fused megakernel grid step for one rank."""
        (
            x_in_ref,
            pos_ref,
            page_table_ref,
            w_refs,
            forced_ref,
            x_out_ref,
            pool_out_ref,
            conv_out_ref,
            rec_out_ref,
            scratch,
        ) = _unpack_kernel_args(args, has_forced_routing=has_forced_routing)

        me = kernel_axis_index(axis_name) if axis_name is not None else jnp.int32(0)
        dma.global_barrier(scratch["b_sem"], me, cfg, axis_name=axis_name, sem_offset=0)
        pos_vec = jnp.asarray([pos_ref[b] for b in range(cfg.B)], dtype=jnp.int32)

        stage_scratch_u8 = cast("TransformedRef", scratch["stage_scratch_u8"])
        wo_buf = cast("jax.Array", stage_plan.view(stage_scratch_u8, wo_slice))
        wout_buf = cast("jax.Array", stage_plan.view(stage_scratch_u8, wout_slice))
        wse2_buf = cast("jax.Array", stage_plan.view(stage_scratch_u8, wse2_slice))

        ar = dma.Qwen38HierAllReduce(
            dma.HierAllReduceBuffers.from_scratch(scratch),
            me,
            cfg,
            axis_name=axis_name,
        )

        x_cur = scratch["x_cur"]
        x_norm = scratch["x_norm"]
        x_cur[...] = x_in_ref[...]

        att_refs = AttentionRefs(
            page_table_ref=page_table_ref,
            pool_out_ref=pool_out_ref,
            w_refs=w_refs,
            scratch=scratch,
            wo_buf=wo_buf,
            me=me,
            axis_name=axis_name,
        )

        gdn_refs = GdnRefs(
            conv_out_ref=conv_out_ref,
            rec_out_ref=rec_out_ref,
            w_refs=w_refs,
            scratch=scratch,
            wout_buf=wout_buf,
        )

        ar_step = 0
        for l_idx in range(cfg.L):
            in_norm_w = w_refs["in_norm"][l_idx, :][None, :]
            x_norm[...] = compute.rmsnorm(x_cur[...], in_norm_w)

            is_att = (l_idx + 1) % cfg.att_every == 0
            if is_att:
                att_idx = l_idx // cfg.att_every
                part_sub = _run_attention_subblock(
                    att_idx,
                    x_norm[...],
                    pos_vec,
                    att_refs,
                    cfg,
                )
            else:
                gdn_idx = l_idx - (l_idx // cfg.att_every)
                part_sub = _run_gdn_subblock(
                    gdn_idx,
                    x_norm[...],
                    gdn_refs,
                    cfg,
                )

            sub_total = ar.allreduce(ar_step & 1, part_sub)
            ar_step += 1
            x_cur[...] = (x_cur[...].astype(jnp.float32) + sub_total).astype(
                jnp.bfloat16
            )

            post_norm_w = w_refs["post_norm"][l_idx, :][None, :]
            x_norm[...] = compute.rmsnorm(x_cur[...], post_norm_w)

            forced_layer = None
            if forced_ref is not None:
                forced_layer = jnp.asarray(
                    [
                        [forced_ref[l_idx, b, k] for k in range(cfg.K)]
                        for b in range(cfg.B)
                    ],
                    dtype=jnp.int32,
                )

            moe_refs = MoeRefs(
                forced_layer=forced_layer,
                w_refs=w_refs,
                scratch=scratch,
                wse2_buf=wse2_buf,
            )
            part_moe = _run_moe_subblock(
                l_idx,
                x_norm[...],
                moe_refs,
                cfg,
            )
            moe_total = ar.allreduce(ar_step & 1, part_moe)
            ar_step += 1
            x_cur[...] = (x_cur[...].astype(jnp.float32) + moe_total).astype(
                jnp.bfloat16
            )

        final_norm_w = w_refs["final_norm"][...][None, :]
        x_out_ref[...] = compute.rmsnorm(x_cur[...], final_norm_w)
        dma.global_barrier(scratch["b_sem"], me, cfg, axis_name=axis_name, sem_offset=2)

    return _kernel


def make_pallas_step_fn(
    cfg: Qwen38Config,
    weights_template: dict[str, jax.Array],
    **options: Unpack[PallasStepOptions],
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Create the per-rank `pl.pallas_call` callable with in-place state aliasing."""
    has_forced_routing = options.get("has_forced_routing", False)
    interpret = options.get("interpret", False)
    axis_name = options.get("axis_name", "tp")

    in_specs, out_specs = plan.make_io_specs(
        weights_template, has_forced_routing=has_forced_routing
    )
    resolved_interpret = (
        pltpu.InterpretParams(detect_races=False, out_of_bounds_reads="raise")
        if interpret is True
        else interpret
    )
    scratch_shapes = plan.make_scratch_specs(cfg, interpret=bool(resolved_interpret))
    kernel_fn = build_qwen38_kernel_fn(
        cfg,
        has_forced_routing=has_forced_routing,
        axis_name=axis_name,
    )

    def _per_rank_step(
        *args: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Execute one per-rank invocation of the megakernel."""
        x_in = args[0]
        pool = args[3]
        conv = args[4]
        rec = args[5]
        out_shape = (
            jax.ShapeDtypeStruct(x_in.shape, x_in.dtype),
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
            jax.ShapeDtypeStruct(conv.shape, conv.dtype),
            jax.ShapeDtypeStruct(rec.shape, rec.dtype),
        )
        call_fn = pl.pallas_call(
            kernel_fn,
            out_shape=out_shape,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(),
                in_specs=in_specs,
                out_specs=out_specs,
                scratch_shapes=[scratch_shapes],
            ),
            input_output_aliases={3: 1, 4: 2, 5: 3},
            interpret=resolved_interpret,
            compiler_params=pltpu.CompilerParams(
                collective_id=0,
                vmem_limit_bytes=plan.vmem_limit_bytes(),
            ),
            name=f"qwen38_2_4t_tp{cfg.R}_dcp{cfg.DCP}_b{cfg.B}",
        )
        return call_fn(*args)

    return _per_rank_step


_STEP_FN_CACHE: dict[
    tuple[object, ...],
    Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
] = {}


def build_sharded_step_fn(
    cfg: Qwen38Config,
    mesh: Mesh,
    weights_template: dict[str, jax.Array],
    **options: Unpack[ShardedStepOptions],
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Wrap `make_pallas_step_fn` in a 32-rank `shard_map` over `mesh`."""
    has_forced_routing = options.get("has_forced_routing", False)
    interpret = options.get("interpret", False)
    axis_name = options.get("axis_name", "tp")

    resolved_interpret = (
        pltpu.InterpretParams(detect_races=False, out_of_bounds_reads="raise")
        if interpret is True
        else interpret
    )
    cache_able = resolved_interpret is False or not bool(
        resolved_interpret.detect_races
    )
    cache_key = (
        cfg,
        tuple(mesh.devices.flat),
        tuple((k, v.shape, v.dtype) for k, v in sorted(weights_template.items())),
        has_forced_routing,
        repr(resolved_interpret),
        axis_name,
    )
    if cache_able and cache_key in _STEP_FN_CACHE:
        return _STEP_FN_CACHE[cache_key]

    per_rank_fn = make_pallas_step_fn(
        cfg,
        weights_template,
        has_forced_routing=has_forced_routing,
        interpret=resolved_interpret,
        axis_name=axis_name,
    )
    w_pspec = {k: P(axis_name) for k in weights_template}
    in_pspecs: list[P | dict[str, P]] = [
        P(axis_name),
        P(axis_name),
        P(axis_name),
        P(axis_name),
        P(axis_name),
        P(axis_name),
        w_pspec,
    ]
    if has_forced_routing:
        in_pspecs.append(P(axis_name))

    def _mapped(
        *args: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Apply squeeze and expand wrappers over rank shards."""
        squeezed = [
            jax.tree_util.tree_map(lambda a: jnp.squeeze(a, axis=0), x) for x in args
        ]
        x_o, p_o, c_o, r_o = per_rank_fn(*squeezed)
        return (
            jnp.expand_dims(x_o, axis=0),
            jnp.expand_dims(p_o, axis=0),
            jnp.expand_dims(c_o, axis=0),
            jnp.expand_dims(r_o, axis=0),
        )

    fn = jax.jit(
        jax.shard_map(
            _mapped,
            mesh=mesh,
            in_specs=tuple(in_pspecs),
            out_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name)),
            check_vma=False,
        )
    )
    if cache_able:
        _STEP_FN_CACHE[cache_key] = fn
    return fn


def pack_gdn_states_into_pools(
    pools_u8: list[jax.Array],
    conv_state: jax.Array,
    rec_state: jax.Array,
    mamba_slots: jax.Array,
    cfg: Qwen38Config,
) -> list[jax.Array]:
    """Pack GDN `rec_state` and `conv_state` into the unified FP8 region pools."""
    out_pools = list(pools_u8)
    has_rank_dim = out_pools[0].ndim == _RANK_POOL_NDIM
    rec_bytes = cfg.VH_R * cfg.KD * cfg.VD * 4
    conv_bytes = (cfg.conv_kernel - 1) * cfg.QKV * 2
    g_per_reg = max(1, cfg.att_every - 1)

    for g_idx in range(cfg.NGDN):
        r_idx = min(g_idx // g_per_reg, len(out_pools) - 1)
        local_g = g_idx % g_per_reg
        pool_reg = out_pools[r_idx]
        for b_idx in range(cfg.B):
            slot_raw = jnp.asarray(mamba_slots[g_idx, b_idx], dtype=jnp.int32)
            slot_g0 = jnp.asarray(
                mamba_slots[(g_idx // g_per_reg) * g_per_reg, b_idx],
                dtype=jnp.int32,
            )
            is_shared_slot = (local_g > 0) & (slot_raw == slot_g0)
            eff_slot = jnp.where(is_shared_slot, slot_raw + local_g * cfg.B, slot_raw)
            valid_slot = slot_raw >= 0
            safe_slot = jnp.where(valid_slot, eff_slot, 0)
            p_base = safe_slot * PAGES_PER_GDN_SLOT
            if has_rank_dim:
                rec_u8 = jax.lax.bitcast_convert_type(
                    rec_state[:, g_idx, b_idx], jnp.uint8
                ).reshape(cfg.R, rec_bytes)
                pages0_3_orig = jax.lax.dynamic_slice(
                    pool_reg,
                    (0, p_base, 0, 0, 0, 0),
                    (cfg.R, 4, 2, cfg.K_R, cfg.D, cfg.page_size),
                ).reshape(cfg.R, -1)
                pages0_3_new = pages0_3_orig.at[:, :rec_bytes].set(rec_u8)
                pages0_3 = jnp.where(valid_slot, pages0_3_new, pages0_3_orig)
                pool_reg = jax.lax.dynamic_update_slice(
                    pool_reg,
                    pages0_3.reshape(cfg.R, 4, 2, cfg.K_R, cfg.D, cfg.page_size),
                    (0, p_base, 0, 0, 0, 0),
                )
                conv_u8 = jax.lax.bitcast_convert_type(
                    conv_state[:, g_idx, b_idx], jnp.uint8
                ).reshape(cfg.R, conv_bytes)
                page4_orig = pool_reg[:, p_base + 4].reshape(cfg.R, -1)
                page4_new = page4_orig.at[:, :conv_bytes].set(conv_u8)
                page4 = jnp.where(valid_slot, page4_new, page4_orig)
                pool_reg = pool_reg.at[:, p_base + 4].set(
                    page4.reshape(cfg.R, 2, cfg.K_R, cfg.D, cfg.page_size)
                )
            else:
                rec_u8 = jax.lax.bitcast_convert_type(
                    rec_state[g_idx, b_idx], jnp.uint8
                ).reshape(rec_bytes)
                pages0_3_orig = jax.lax.dynamic_slice(
                    pool_reg,
                    (p_base, 0, 0, 0, 0),
                    (4, 2, cfg.K_R, cfg.D, cfg.page_size),
                ).reshape(-1)
                pages0_3_new = pages0_3_orig.at[:rec_bytes].set(rec_u8)
                pages0_3 = jnp.where(valid_slot, pages0_3_new, pages0_3_orig)
                pool_reg = jax.lax.dynamic_update_slice(
                    pool_reg,
                    pages0_3.reshape(4, 2, cfg.K_R, cfg.D, cfg.page_size),
                    (p_base, 0, 0, 0, 0),
                )
                conv_u8 = jax.lax.bitcast_convert_type(
                    conv_state[g_idx, b_idx], jnp.uint8
                ).reshape(conv_bytes)
                page4_orig = pool_reg[p_base + 4].reshape(-1)
                page4_new = page4_orig.at[:conv_bytes].set(conv_u8)
                page4 = jnp.where(valid_slot, page4_new, page4_orig)
                pool_reg = pool_reg.at[p_base + 4].set(
                    page4.reshape(2, cfg.K_R, cfg.D, cfg.page_size)
                )
        out_pools[r_idx] = pool_reg
    return out_pools


def unpack_gdn_states_from_pools(
    pools_u8: list[jax.Array],
    mamba_slots: jax.Array,
    cfg: Qwen38Config,
) -> tuple[jax.Array, jax.Array]:
    """Unpack GDN `conv_state` and `rec_state` from the unified FP8 region pools."""
    has_rank_dim = pools_u8[0].ndim == _RANK_POOL_NDIM
    rec_bytes = cfg.VH_R * cfg.KD * cfg.VD * 4
    conv_bytes = (cfg.conv_kernel - 1) * cfg.QKV * 2
    g_per_reg = max(1, cfg.att_every - 1)

    conv_layers = []
    rec_layers = []
    for g_idx in range(cfg.NGDN):
        r_idx = min(g_idx // g_per_reg, len(pools_u8) - 1)
        local_g = g_idx % g_per_reg
        pool_reg = pools_u8[r_idx]

        conv_b = []
        rec_b = []
        for b_idx in range(cfg.B):
            slot_raw = jnp.asarray(mamba_slots[g_idx, b_idx], dtype=jnp.int32)
            slot_g0 = jnp.asarray(
                mamba_slots[(g_idx // g_per_reg) * g_per_reg, b_idx],
                dtype=jnp.int32,
            )
            is_shared_slot = (local_g > 0) & (slot_raw == slot_g0)
            eff_slot = jnp.where(is_shared_slot, slot_raw + local_g * cfg.B, slot_raw)
            valid_slot = slot_raw >= 0
            safe_slot = jnp.where(valid_slot, eff_slot, 0)
            p_base = safe_slot * PAGES_PER_GDN_SLOT

            if has_rank_dim:
                rec_raw = (
                    jax.lax.dynamic_slice(
                        pool_reg,
                        (0, p_base, 0, 0, 0, 0),
                        (cfg.R, 4, 2, cfg.K_R, cfg.D, cfg.page_size),
                    )
                    .reshape(cfg.R, -1)[:, :rec_bytes]
                    .reshape(cfg.R, cfg.VH_R, cfg.KD, cfg.VD, 4)
                )
                rec_val = jax.lax.bitcast_convert_type(rec_raw, jnp.float32)
                rec_zero = jnp.zeros_like(rec_val)
                rec_b.append(jnp.where(valid_slot, rec_val, rec_zero))

                conv_raw = (
                    pool_reg[:, p_base + 4]
                    .reshape(cfg.R, -1)[:, :conv_bytes]
                    .reshape(cfg.R, cfg.conv_kernel - 1, cfg.QKV, 2)
                )
                conv_val = jax.lax.bitcast_convert_type(conv_raw, jnp.bfloat16)
                conv_zero = jnp.zeros_like(conv_val)
                conv_b.append(jnp.where(valid_slot, conv_val, conv_zero))
            else:
                rec_raw = (
                    jax.lax.dynamic_slice(
                        pool_reg,
                        (p_base, 0, 0, 0, 0),
                        (4, 2, cfg.K_R, cfg.D, cfg.page_size),
                    )
                    .reshape(-1)[:rec_bytes]
                    .reshape(cfg.VH_R, cfg.KD, cfg.VD, 4)
                )
                rec_val = jax.lax.bitcast_convert_type(rec_raw, jnp.float32)
                rec_zero = jnp.zeros_like(rec_val)
                rec_b.append(jnp.where(valid_slot, rec_val, rec_zero))

                conv_raw = (
                    pool_reg[p_base + 4]
                    .reshape(-1)[:conv_bytes]
                    .reshape(cfg.conv_kernel - 1, cfg.QKV, 2)
                )
                conv_val = jax.lax.bitcast_convert_type(conv_raw, jnp.bfloat16)
                conv_zero = jnp.zeros_like(conv_val)
                conv_b.append(jnp.where(valid_slot, conv_val, conv_zero))

        ax_b = 1 if has_rank_dim else 0
        conv_layers.append(jnp.stack(conv_b, axis=ax_b))
        rec_layers.append(jnp.stack(rec_b, axis=ax_b))
    ax_g = 1 if has_rank_dim else 0
    return jnp.stack(conv_layers, axis=ax_g), jnp.stack(rec_layers, axis=ax_g)


def pack_gdn_states_into_unified_pool(
    pool_u8: jax.Array,
    conv_state: jax.Array,
    rec_state: jax.Array,
    mamba_slots: jax.Array,
    cfg: Qwen38Config,
) -> jax.Array:
    """Pack GDN `rec_state` and `conv_state` into a contiguous 6D `(NATT, ...)` pool."""
    rec_bytes = cfg.VH_R * cfg.KD * cfg.VD * 4
    conv_bytes = (cfg.conv_kernel - 1) * cfg.QKV * 2
    g_per_reg = max(1, cfg.att_every - 1)
    n_regs = int(pool_u8.shape[0])

    for g_idx in range(cfg.NGDN):
        r_idx = min(g_idx // g_per_reg, n_regs - 1)
        local_g = g_idx % g_per_reg
        for b_idx in range(cfg.B):
            slot_raw = jnp.asarray(mamba_slots[g_idx, b_idx], dtype=jnp.int32)
            slot_g0 = jnp.asarray(
                mamba_slots[(g_idx // g_per_reg) * g_per_reg, b_idx],
                dtype=jnp.int32,
            )
            is_shared_slot = (local_g > 0) & (slot_raw == slot_g0)
            eff_slot = jnp.where(is_shared_slot, slot_raw + local_g * cfg.B, slot_raw)
            valid_slot = slot_raw >= 0
            safe_slot = jnp.where(valid_slot, eff_slot, 0)
            p_base = safe_slot * PAGES_PER_GDN_SLOT

            rec_u8 = jax.lax.bitcast_convert_type(
                rec_state[g_idx, b_idx], jnp.uint8
            ).reshape(rec_bytes)
            pages0_3_orig = jax.lax.dynamic_slice(
                pool_u8,
                (r_idx, p_base, 0, 0, 0, 0),
                (1, 4, 2, cfg.K_R, cfg.D, cfg.page_size),
            ).reshape(-1)
            pages0_3_new = pages0_3_orig.at[:rec_bytes].set(rec_u8)
            pages0_3 = jnp.where(valid_slot, pages0_3_new, pages0_3_orig)
            pool_u8 = jax.lax.dynamic_update_slice(
                pool_u8,
                pages0_3.reshape(1, 4, 2, cfg.K_R, cfg.D, cfg.page_size),
                (r_idx, p_base, 0, 0, 0, 0),
            )

            conv_u8 = jax.lax.bitcast_convert_type(
                conv_state[g_idx, b_idx], jnp.uint8
            ).reshape(conv_bytes)
            page4_orig = jax.lax.dynamic_slice(
                pool_u8,
                (r_idx, p_base + 4, 0, 0, 0, 0),
                (1, 1, 2, cfg.K_R, cfg.D, cfg.page_size),
            ).reshape(-1)
            page4_new = page4_orig.at[:conv_bytes].set(conv_u8)
            page4 = jnp.where(valid_slot, page4_new, page4_orig)
            pool_u8 = jax.lax.dynamic_update_slice(
                pool_u8,
                page4.reshape(1, 1, 2, cfg.K_R, cfg.D, cfg.page_size),
                (r_idx, p_base + 4, 0, 0, 0, 0),
            )
    return pool_u8


def unpack_gdn_states_from_unified_pool(
    pool_u8: jax.Array,
    mamba_slots: jax.Array,
    cfg: Qwen38Config,
) -> tuple[jax.Array, jax.Array]:
    """Unpack GDN `conv_state` and `rec_state` from a 6D `(NATT, ...)` pool."""
    rec_bytes = cfg.VH_R * cfg.KD * cfg.VD * 4
    conv_bytes = (cfg.conv_kernel - 1) * cfg.QKV * 2
    g_per_reg = max(1, cfg.att_every - 1)
    n_regs = int(pool_u8.shape[0])

    conv_layers = []
    rec_layers = []
    for g_idx in range(cfg.NGDN):
        r_idx = min(g_idx // g_per_reg, n_regs - 1)
        local_g = g_idx % g_per_reg
        conv_b = []
        rec_b = []
        for b_idx in range(cfg.B):
            slot_raw = jnp.asarray(mamba_slots[g_idx, b_idx], dtype=jnp.int32)
            slot_g0 = jnp.asarray(
                mamba_slots[(g_idx // g_per_reg) * g_per_reg, b_idx],
                dtype=jnp.int32,
            )
            is_shared_slot = (local_g > 0) & (slot_raw == slot_g0)
            eff_slot = jnp.where(is_shared_slot, slot_raw + local_g * cfg.B, slot_raw)
            valid_slot = slot_raw >= 0
            safe_slot = jnp.where(valid_slot, eff_slot, 0)
            p_base = safe_slot * PAGES_PER_GDN_SLOT

            rec_raw = (
                jax.lax.dynamic_slice(
                    pool_u8,
                    (r_idx, p_base, 0, 0, 0, 0),
                    (1, 4, 2, cfg.K_R, cfg.D, cfg.page_size),
                )
                .reshape(-1)[:rec_bytes]
                .reshape(cfg.VH_R, cfg.KD, cfg.VD, 4)
            )
            rec_val = jax.lax.bitcast_convert_type(rec_raw, jnp.float32)
            rec_b.append(jnp.where(valid_slot, rec_val, jnp.zeros_like(rec_val)))

            conv_raw = (
                jax.lax.dynamic_slice(
                    pool_u8,
                    (r_idx, p_base + 4, 0, 0, 0, 0),
                    (1, 1, 2, cfg.K_R, cfg.D, cfg.page_size),
                )
                .reshape(-1)[:conv_bytes]
                .reshape(cfg.conv_kernel - 1, cfg.QKV, 2)
            )
            conv_val = jax.lax.bitcast_convert_type(conv_raw, jnp.bfloat16)
            conv_b.append(jnp.where(valid_slot, conv_val, jnp.zeros_like(conv_val)))

        conv_layers.append(jnp.stack(conv_b, axis=0))
        rec_layers.append(jnp.stack(rec_b, axis=0))
    return jnp.stack(conv_layers, axis=0), jnp.stack(rec_layers, axis=0)


def _adapt_pool_for_kernel(
    pool: jax.Array, cfg: Qwen38Config
) -> tuple[jax.Array, bool]:
    """Ensure `pool` has `(..., NATT, num_pages, 2, K_R, D, page_size)` layout."""
    if pool.ndim == _UNRANKED_POOL_NDIM:
        if cfg.NATT == 1:
            return jnp.expand_dims(pool, axis=0), True
        return jnp.repeat(pool[None, ...], cfg.NATT, axis=0), True
    if pool.ndim == _RANK_POOL_NDIM and pool.shape[0] == cfg.R:
        if cfg.NATT == 1:
            return jnp.expand_dims(pool, axis=1), True
        if cfg.R != cfg.NATT:
            return jnp.repeat(pool[:, None, ...], cfg.NATT, axis=1), True
    return pool, False


def _restore_pool_from_kernel(
    pool_k: jax.Array,
    *,
    added_natt: bool,
) -> jax.Array:
    """Restore original pool dimensions after kernel execution."""
    if not added_natt:
        return pool_k
    return pool_k[:, 0] if pool_k.ndim == _POOL_7D_NDIM else pool_k[0]


def _resolve_step_mesh_and_interpret(
    cfg: Qwen38Config,
    mesh: Mesh | None,
    *,
    interpret: bool | pltpu.InterpretParams | None,
) -> tuple[Mesh, bool | pltpu.InterpretParams]:
    """Resolve a valid `cfg.R` device mesh and default interpret flag."""
    if mesh is None or not issubclass(type(mesh), Mesh) or len(mesh.devices) != cfg.R:
        devices = jax.devices()
        if len(devices) < cfg.R:
            err_msg = (
                f"Need at least {cfg.R} JAX devices for R={cfg.R}, found {len(devices)}"
            )
            raise MegakernelDeviceError(err_msg)
        mesh = Mesh(devices[: cfg.R], ("tp",))
    resolved_interpret = (
        mesh.devices.flat[0].platform == "cpu" if interpret is None else interpret
    )
    return mesh, resolved_interpret


def run_qwen38_dcp8_step(
    problem: Mapping[str, jax.Array | dict[str, jax.Array]],
    cfg: Qwen38Config,
    **options: Unpack[StepOptions],
) -> dict[str, jax.Array]:
    """Execute one 32-rank TP32xDCP8 step on CPU interpret or TPU devices."""
    use_forced_routing = options.get("use_forced_routing", True)
    return_kv_updates = options.get("return_kv_updates", True)
    mesh, interpret = _resolve_step_mesh_and_interpret(
        cfg, options.get("mesh"), interpret=options.get("interpret")
    )

    weights_val = problem["weights"]
    if not isinstance(weights_val, dict):
        msg = "problem['weights'] must be a dict[str, jax.Array]"
        raise TypeError(msg)
    norm_weights = (
        plan.normalize_weights_for_pallas(weights_val, cfg)
        if weights_val["wqkv"].ndim != _NORM_WQKV_NDIM
        else weights_val
    )
    per_rank_w_tmpl = {k: v[0] for k, v in norm_weights.items()}

    raw_pool = cast("jax.Array", problem["pool"])
    pool_in, added_natt = _adapt_pool_for_kernel(raw_pool, cfg)
    step_fn = build_sharded_step_fn(
        cfg,
        mesh,
        per_rank_w_tmpl,
        has_forced_routing=use_forced_routing,
        interpret=interpret,
    )

    sharding = jax.sharding.NamedSharding(mesh, P("tp"))
    is_cpu = mesh.devices.flat[0].platform == "cpu"

    def _put(arr: jax.Array) -> jax.Array:
        """Place array onto device mesh according to sharding."""
        if getattr(arr, "sharding", None) == sharding:
            return arr
        if is_cpu:
            return jax.device_put(np.asarray(arr), sharding)
        return jax.device_put(arr, sharding)

    x_s = _put(cast("jax.Array", problem["x"]))
    pos_arr = cast("jax.Array", problem["pos"])
    pos_s = _put(pos_arr)
    pt_s = _put(cast("jax.Array", problem["page_table"]))
    pool_s = _put(pool_in)
    conv_s = _put(cast("jax.Array", problem["conv"]))
    rec_s = _put(cast("jax.Array", problem["rec"]))
    w_s = {k: _put(v) for k, v in norm_weights.items()}

    if use_forced_routing:
        forced_top = cast("jax.Array", problem["forced_top_ids"])
        forced_s = _put(forced_top)
        x_out, pool_out, conv_out, rec_out = step_fn(
            x_s, pos_s, pt_s, pool_s, conv_s, rec_s, w_s, forced_s
        )
    else:
        x_out, pool_out, conv_out, rec_out = step_fn(
            x_s, pos_s, pt_s, pool_s, conv_s, rec_s, w_s
        )

    out: dict[str, jax.Array] = {
        "x": x_out,
        "conv": conv_out,
        "rec": rec_out,
    }
    if return_kv_updates:
        out["pool"] = _restore_pool_from_kernel(pool_out, added_natt=added_natt)
    return out
