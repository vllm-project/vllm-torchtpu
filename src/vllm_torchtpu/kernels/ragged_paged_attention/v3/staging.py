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
"""Shared Pallas Q/O & KV DMA staging and VMEM tile load/store for RPA v3."""

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.pallas_util import (
    async_copy,
    broadcast_minor,
    cdiv,
    strided_load,
    strided_store,
)

# Row indices into the `(4, 2)` double-buffered DMA semaphore array `sems`.
SEM_BKV_FETCH = 0
SEM_BQ_FETCH = 1
SEM_BO_SEND = 2
SEM_BKV_UPDATE = 3


class StagingRefs(NamedTuple):
    """Double-buffered Q/O HBM, VMEM, and ring-tracking references."""

    q_hbm_ref: Any
    o_hbm_ref: Any
    bq_x2_ref: Any
    bo_x2_ref: Any
    cu_q_lens_ref: Any
    bo_ids_ref: Any
    sems: Any


@dataclasses.dataclass(frozen=True)
class KvUpdateConfig:
    """Static block/page geometry for `update_kv_cache`."""

    page_size: int
    pages_per_seq: int
    bkv_sz: int
    bkv_p: int
    debug_mode: bool = False


class KvUpdateRing(NamedTuple):
    """Double-buffered KV-cache writeback ring tracker."""

    bkv_update_ids_ref: Any
    update_fn: Callable[..., Any]

    def start(self, seq_idx, bkv_sem_idx, offset, update_sz) -> None:
        self.bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        self.bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        self.bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        self.update_fn(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait(self, bkv_sem_idx) -> None:
        update_sz = self.bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = self.bkv_update_ids_ref[bkv_sem_idx]
            offset = self.bkv_update_ids_ref[bkv_sem_idx + 2]
            self.bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            self.update_fn(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)


def update_kv_cache(
    seq_idx,
    bkv_sem_idx,
    offset,
    update_sz,
    *,
    vmem_ref,
    updated_kv_cache_hbm_ref,
    page_indices_ref,
    sems,
    cfg: KvUpdateConfig,
    wait: bool = False,
) -> None:
    """Start or wait on page-wise async DMA writeback from `vmem_ref` into HBM."""
    sem = sems.at[SEM_BKV_UPDATE, bkv_sem_idx]
    bkv_id = offset // cfg.bkv_sz
    kv_p_start = offset // cfg.page_size
    kv_p_end = cdiv(offset + update_sz, cfg.page_size)
    ignore = offset % cfg.page_size
    p_ignore = kv_p_start - bkv_id * cfg.bkv_p
    page_indices_offset = seq_idx * cfg.pages_per_seq + kv_p_start

    cache_hbm_shape = updated_kv_cache_hbm_ref.shape
    cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(
        cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:]
    )

    if cfg.debug_mode:
        pl.debug_print(
            "[RPA debug]"
            f" -----------{'wait' if wait else 'start'}_update_kv_cache-----------"
        )
        pl.debug_print("[RPA debug] seq_idx={}", seq_idx)
        pl.debug_print("[RPA debug] bkv_sem_idx={}", bkv_sem_idx)
        pl.debug_print("[RPA debug] offset={}", offset)
        pl.debug_print("[RPA debug] update_sz={}", update_sz)
        pl.debug_print("[RPA debug] bkv_id={}", bkv_id)
        pl.debug_print("[RPA debug] kv_p_start={}", kv_p_start)
        pl.debug_print("[RPA debug] kv_p_end={}", kv_p_end)
        pl.debug_print("[RPA debug] ignore={}", ignore)
        pl.debug_print("[RPA debug] p_ignore={}", p_ignore)
        pl.debug_print("[RPA debug] page_indices_offset={}", page_indices_offset)

    if not wait:

        def loop_body(i, states):
            cur_sz, cur_ignore = states
            sz = jnp.minimum(cfg.page_size - cur_ignore, cur_sz)
            async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * cfg.page_size + cur_ignore, sz)],
                cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * cfg.page_size
                        + cur_ignore,
                        sz,
                    )
                ],
                sem,
                wait=False,
                debug_mode=cfg.debug_mode,
            )
            if cfg.debug_mode:
                pl.debug_print("[RPA debug] loop_body i={}, sz={}", i, sz)
            return cur_sz - sz, 0

        lax.fori_loop(
            0,
            kv_p_end - kv_p_start,
            loop_body,
            (update_sz, ignore),  # total transfer size
            unroll=False,
        )
    else:
        dst = cache_hbm_ref.at[pl.ds(0, update_sz)]
        async_copy(
            src=dst,
            dst=dst,
            sem=sem,
            wait=True,
            debug_mode=cfg.debug_mode,
        )


def _transfer_qo(
    refs: StagingRefs,
    seq_idx,
    blk_idx,
    sem_idx,
    *,
    bq_sz: int,
    is_send: bool,
    wait: bool = False,
    debug_mode: bool = False,
) -> None:
    sem = refs.sems.at[SEM_BO_SEND if is_send else SEM_BQ_FETCH, sem_idx]
    vmem_ref = (refs.bo_x2_ref if is_send else refs.bq_x2_ref).at[sem_idx]
    q_len_start = refs.cu_q_lens_ref[seq_idx] + blk_idx * bq_sz
    q_end = refs.cu_q_lens_ref[seq_idx + 1]
    sz = jnp.minimum(bq_sz, q_end - q_len_start)
    if debug_mode:
        op = "send_bo" if is_send else "fetch_bq"
        idx_name = "bo_idx" if is_send else "bq_idx"
        sem_name = "bo_sem_idx" if is_send else "bq_sem_idx"
        pl.debug_print(
            f"[RPA debug] -----------{'wait' if wait else 'start'}_{op}-----------"
        )
        pl.debug_print("[RPA debug] seq_idx={}", seq_idx)
        pl.debug_print(f"[RPA debug] {idx_name}={{}}", blk_idx)
        pl.debug_print(f"[RPA debug] {sem_name}={{}}", sem_idx)
        pl.debug_print("[RPA debug] q_len_start={}", q_len_start)
        pl.debug_print("[RPA debug] q_end={}", q_end)
        pl.debug_print("[RPA debug] sz={}", sz)
    vmem_slice = vmem_ref.at[:, pl.ds(0, sz)]
    hbm_slice = (refs.o_hbm_ref if is_send else refs.q_hbm_ref).at[
        :, pl.ds(q_len_start, sz)
    ]
    src, dst = (vmem_slice, hbm_slice) if is_send else (hbm_slice, vmem_slice)
    async_copy(src, dst, sem, wait, debug_mode=debug_mode)


def fetch_bq(
    refs: StagingRefs,
    seq_idx,
    bq_idx,
    bq_sem_idx,
    *,
    bq_sz: int,
    wait: bool = False,
    debug_mode: bool = False,
) -> None:
    _transfer_qo(
        refs,
        seq_idx,
        bq_idx,
        bq_sem_idx,
        bq_sz=bq_sz,
        is_send=False,
        wait=wait,
        debug_mode=debug_mode,
    )


def send_bo(
    refs: StagingRefs,
    seq_idx,
    bo_idx,
    bo_sem_idx,
    *,
    bq_sz: int,
    wait: bool = False,
    debug_mode: bool = False,
) -> None:
    _transfer_qo(
        refs,
        seq_idx,
        bo_idx,
        bo_sem_idx,
        bq_sz=bq_sz,
        is_send=True,
        wait=wait,
        debug_mode=debug_mode,
    )


def start_send_bo(
    refs: StagingRefs,
    seq_idx,
    bo_idx,
    bo_sem_idx,
    *,
    bq_sz: int,
    debug_mode: bool = False,
) -> None:
    refs.bo_ids_ref[bo_sem_idx] = seq_idx
    refs.bo_ids_ref[bo_sem_idx + 2] = bo_idx
    send_bo(refs, seq_idx, bo_idx, bo_sem_idx, bq_sz=bq_sz, debug_mode=debug_mode)


def wait_send_bo(
    refs: StagingRefs,
    start_seq_idx,
    seq_idx,
    bo_sem_idx,
    *,
    bq_sz: int,
    debug_mode: bool = False,
) -> None:
    old_seq_idx = refs.bo_ids_ref[bo_sem_idx]
    old_bo_idx = refs.bo_ids_ref[bo_sem_idx + 2]

    @pl.when(jnp.logical_and(start_seq_idx <= old_seq_idx, old_seq_idx <= seq_idx))
    def _():
        send_bo(
            refs,
            old_seq_idx,
            old_bo_idx,
            bo_sem_idx,
            bq_sz=bq_sz,
            wait=True,
            debug_mode=debug_mode,
        )


def load_bq(bq_x2_ref, bq_sem_idx, kv_head_idx, start, sz):
    """Load `sz` query tokens for `kv_head_idx` from double-buffered `bq_x2_ref`."""
    _, _, bq_sz, nq_per_kv_packed, _, head_dim = bq_x2_ref.shape
    q_ref = (
        bq_x2_ref.bitcast(jnp.uint32)
        .at[bq_sem_idx, kv_head_idx]
        .reshape(bq_sz * nq_per_kv_packed, head_dim)
    )
    return strided_load(
        q_ref,
        start * nq_per_kv_packed,
        sz * nq_per_kv_packed,
        1,
        dtype=bq_x2_ref.dtype,
    )


def load_bkv(bkv_x2_ref, bkv_sem_idx, kv_head_idx, start, sz):
    """Load `(k, v)` tiles of `sz` tokens for `kv_head_idx` from `bkv_x2_ref`."""
    _, bkv_sz, bkv_stride, kv_packing, head_dim = bkv_x2_ref.shape
    kv_dtype = bkv_x2_ref.dtype
    start *= bkv_stride
    sz *= bkv_stride
    kv_ref = (
        bkv_x2_ref.bitcast(jnp.uint32)
        .at[bkv_sem_idx]
        .reshape(bkv_sz * bkv_stride, head_dim)
    )
    if kv_packing == 1:
        start += kv_head_idx * 2
        return (
            strided_load(kv_ref, start, sz, bkv_stride, dtype=kv_dtype),
            strided_load(kv_ref, start + 1, sz, bkv_stride, dtype=kv_dtype),
        )

    num_kv_per_load = kv_packing // 2
    kv = strided_load(kv_ref, start + kv_head_idx // num_kv_per_load, sz, bkv_stride)
    bitwidth = 32 // kv_packing
    repack_ty = jnp.dtype(f"uint{bitwidth}")
    k = kv >> ((kv_head_idx % num_kv_per_load) * 2 * bitwidth)
    v = k >> bitwidth
    return (
        pltpu.bitcast(k.astype(repack_ty), kv_dtype),
        pltpu.bitcast(v.astype(repack_ty), kv_dtype),
    )


def normalize_bo(acc_ref, l_ref, *, out_dtype):
    """Normalize `acc_ref` by `l_ref` and cast to `out_dtype`."""
    acc = acc_ref[...]
    denom = broadcast_minor(l_ref[...], acc.shape)
    return (
        acc * pltpu.reciprocal(denom, approx=True)
        if (denom.dtype == jnp.float32 and out_dtype != jnp.float32)
        else lax.div(acc, denom)
    ).astype(out_dtype)


def store_bo(bo_x2_ref, bo_sem_idx, out) -> None:
    """Store normalized `out` into `bo_x2_ref[bo_sem_idx]`."""
    assert out.dtype == bo_x2_ref.dtype
    _, actual_num_kv_heads, bq_sz, nq_per_kv_packed, _, head_dim = bo_x2_ref.shape
    out_ref = (
        bo_x2_ref.at[bo_sem_idx]
        .bitcast(jnp.int32)
        .reshape(actual_num_kv_heads * bq_sz * nq_per_kv_packed, head_dim)
    )
    out_i32 = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
    strided_store(out_ref, 0, out_ref.shape[0], 1, out_i32)
