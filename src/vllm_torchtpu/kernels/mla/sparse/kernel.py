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
"""TPU-Friendly MLA Ragged Paged Attention kernel - GLM5.2 DeepSeek Sparse Attention."""

import functools
import logging

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.sparse import dsa_gather, native_sc_gather

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
TENSORCORE_CACHE_LAYOUT = "tensorcore"
SPARSECORE_CACHE_LAYOUT = "sparsecore"

# A plain child of vLLM's "vllm" logger -- the name
# vllm_torchtpu.logger.init_logger(__name__) resolves to -- so these
# trace-time lines reach vllm.log without importing vLLM in a kernel file.
logger = logging.getLogger("vllm." + __name__)
_LOGGED_PATCHES: set = set()


def _log_patch_once(msg: str, *args) -> None:
    """`logger.info` once per (msg, args) per process."""
    key = (msg, *args)
    if key in _LOGGED_PATCHES:
        return
    _LOGGED_PATCHES.add(key)
    logger.info(msg, *args)


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def _largest_divisor(x: int, cap: int) -> int:
    """Largest divisor of ``x`` that is <= ``cap``."""
    for candidate in range(min(x, cap), 0, -1):
        if x % candidate == 0:
            return candidate
    return 1


_RAGGED_K_RUNGS: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)


def get_dcp_kv_slice_rungs(bkv_sz: int) -> tuple[int, ...]:
    """Return the active 64-aligned dynamic KV slice ladder rungs for ``bkv_sz``.

    For ``bkv_sz = 2048`` (full ragged-safe DCP width with ``SLOT_CAP = 0``),
    returns the 6-rung ladder ``(64, 128, 256, 512, 1024, 2048)``
    (``thirty_second_sz, sixteenth_sz, eighth_sz, quarter_sz, half_sz, bkv_sz``).
    Every rung >= 64 and divisible by 64 preserves 16-row (``64 // ROPE_PACKING``)
    SparseCore ROPE DMA sublane alignment and 64-column MXU contracting alignment.
    """
    thirty_second_sz = bkv_sz // 32
    sixteenth_sz = bkv_sz // 16
    eighth_sz = bkv_sz // 8
    quarter_sz = bkv_sz // 4
    half_sz = bkv_sz // 2
    if thirty_second_sz >= 64 and thirty_second_sz % 64 == 0 and bkv_sz % 32 == 0:
        return (
            thirty_second_sz,
            sixteenth_sz,
            eighth_sz,
            quarter_sz,
            half_sz,
            bkv_sz,
        )
    if sixteenth_sz >= 64 and sixteenth_sz % 64 == 0 and bkv_sz % 16 == 0:
        return (sixteenth_sz, eighth_sz, quarter_sz, half_sz, bkv_sz)
    if eighth_sz >= 64 and eighth_sz % 64 == 0 and bkv_sz % 8 == 0:
        return (eighth_sz, quarter_sz, half_sz, bkv_sz)
    if quarter_sz >= 64 and quarter_sz % 64 == 0 and bkv_sz % 4 == 0:
        return (quarter_sz, half_sz, bkv_sz)
    if half_sz >= 64 and half_sz % 64 == 0 and bkv_sz % 2 == 0:
        return (half_sz, bkv_sz)
    return (bkv_sz,)


def select_dcp_kv_slice_size(bkv_sz: int, batch_max_kv_len: int) -> int:
    """Select the active dynamic KV slice rung (or 0 for empty-batch bypass)."""
    if batch_max_kv_len <= 0:
        return 0
    for rung in get_dcp_kv_slice_rungs(bkv_sz):
        if batch_max_kv_len <= rung:
            return rung
    return bkv_sz


def get_dcp_gather_atom_rungs(steps_per_window: int) -> tuple[int, ...]:
    """Return sorted power-of-2 step/atom counts in ``[1, steps_per_window]``."""
    if steps_per_window <= 0:
        return ()
    rungs: list[int] = []
    r = 1
    while r < steps_per_window:
        rungs.append(r)
        r *= 2
    rungs.append(steps_per_window)
    return tuple(rungs)


def compute_dcp_gather_rung_triggers(
    topk: int,
    gather_step_size: int,
    steps_per_window: int,
) -> jax.Array:
    """Return per-column ``kv_len`` threshold at or below which ``SKIP_INDEX`` is stamped.

    Within each ``gather_window_size = steps_per_window * gather_step_size``
    outer batch, step ``in_win_step`` belongs to the smallest power-of-2 rung
    ``r_steps`` in ``get_dcp_gather_atom_rungs(steps_per_window)`` with
    ``in_win_step < r_steps``. That rung executes iff
    ``kv_len > win_base + prev_steps * gather_step_size`` (where ``prev_steps``
    is the preceding rung, or ``0`` for the first rung ``r_steps = 1``).
    Thus stamping ``SKIP_INDEX = -1`` whenever
    ``win_base + prev_steps * gather_step_size >= kv_len`` guarantees:
      1. Every step ``< r_active`` in the active power-of-2 rung retains valid
         non-negative scattered physical indices (never ``-1`` for ``pl.Indirect``).
      2. Step ``r_active`` (the boundary check read by ``native_sc_gather`` /
         ``dsa_gather``) is ``-1``, so all steps ``>= r_active`` are skipped.
    """
    gather_window_size = steps_per_window * gather_step_size
    cols = jnp.arange(topk, dtype=jnp.int32)
    win_base = (cols // gather_window_size) * gather_window_size
    in_win_step = (cols - win_base) // gather_step_size
    rungs = get_dcp_gather_atom_rungs(steps_per_window)
    prev_steps = jnp.zeros_like(in_win_step)
    for idx in range(1, len(rungs)):
        lo = rungs[idx - 1]
        hi = rungs[idx]
        prev_steps = jnp.where(
            (in_win_step >= lo) & (in_win_step < hi),
            lo,
            prev_steps,
        )
    return win_base + prev_steps * gather_step_size


_GATHER_PAGE_CHUNK = 128


def _gather_page_ids_kernel(windows_ref, logical_ref, out_ref, *, num_chunks):
    logical = logical_ref[...]  # i32[block_tokens, topk]
    out = jnp.zeros_like(logical)
    for c in range(num_chunks):
        window_chunk = windows_ref[
            :, c * _GATHER_PAGE_CHUNK : (c + 1) * _GATHER_PAGE_CHUNK
        ]  # i32[block_tokens, 128]
        local = logical - c * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(
            window_chunk, jnp.clip(local, 0, _GATHER_PAGE_CHUNK - 1), axis=1
        )
        out = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered, out)
    out_ref[...] = out


def gather_page_ids(
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_page_ids: jax.Array,  # i32[num_tokens, topk]  (logical page within seq)
    seq_ids_segment: jax.Array,  # i32[num_tokens]  (token -> seq id)
    max_num_seqs: int,
    *,
    block_tokens: int = 8,
) -> jax.Array:
    """Gathers physical page ids for the CSA top-k tokens."""
    num_tokens, topk = seq_page_ids.shape
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    num_chunks = cdiv(pages_per_seq, _GATHER_PAGE_CHUNK)
    padded_pps = num_chunks * _GATHER_PAGE_CHUNK

    page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
    if padded_pps != pages_per_seq:
        page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
    # Per-token page-table window. This is a whole-row gather.
    windows = page_table[seq_ids_segment]  # i32[num_tokens, padded_pps]
    logical = jnp.clip(seq_page_ids, 0, pages_per_seq - 1)

    padded_tokens = align_to(num_tokens, block_tokens)
    if padded_tokens != num_tokens:
        pad = padded_tokens - num_tokens
        windows = jnp.pad(windows, ((0, pad), (0, 0)))
        logical = jnp.pad(logical, ((0, pad), (0, 0)))

    out = pl.pallas_call(
        functools.partial(_gather_page_ids_kernel, num_chunks=num_chunks),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, padded_pps), lambda t: (t, 0)),
                pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            ],
            out_specs=pl.BlockSpec((block_tokens, topk), lambda t: (t, 0)),
            grid=(padded_tokens // block_tokens,),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
        ),
        name="gather_page_ids",
    )(windows, logical)
    return out[:num_tokens]


def _attention_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_tokens] valid top-k count per query token
    start_end_seq_idx_ref,  # [2] (start_seq_idx, end_seq_idx)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    # Input
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # [num_tokens, topk * WORD_BYTES, TILE_LANE_BYTES] (gathered, raw tiles)
    cache_kv_nope_hbm_ref,
    cache_kv_rope_hbm_ref,  # [num_tokens, topk, rope_dim] (gathered)
    # Output (+ scratch) refs are positional and their count depends on
    # `return_lse`, so they arrive packed and are unpacked below.
    *refs,
    sm_scale: float,
    k_scale: float = 1.0,
    batch_size: int = 1,
    return_lse: bool = False,
    sparsecore_layout: bool = False,
    is_dcp_mode: bool = False,
    split_q: bool = False,
    lse_lane_dense: bool = False,
):
    q_pe_hbm_ref = None
    bq_pe_x2_ref = None
    if split_q:
        q_pe_hbm_ref = refs[0]  # [max_num_tokens, num_q_heads, rope_dim]
        bq_pe_x2_ref = refs[-1]  # [2, bsz, num_q_heads, rope_dim]
        refs = refs[1:-1]
    if return_lse:
        (
            _lse_in_hbm_ref,  # aliased with lse_hbm_ref; never read
            o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
            lse_hbm_ref,  # f32[max_num_tokens, num_q_heads, 1] or f32[max_num_tokens, num_q_heads]
            bkv_nope_x2_ref,  # [2, bsz, topk * WORD_BYTES, TILE_LANE_BYTES]
            bkv_rope_x2_ref,  # [2, bsz, topk, rope_dim]
            bq_x2_ref,  # [2, bsz, num_q_heads, head_dim]
            bo_x2_ref,  # [2, bsz, num_q_heads, head_dim]
            blse_x2_ref,  # f32[2, bsz, num_q_heads, 1] or f32[2, bsz, num_q_heads]
            sems,  # [5, 2]
        ) = refs
    else:
        (
            o_hbm_ref,
            bkv_nope_x2_ref,
            bkv_rope_x2_ref,
            bq_x2_ref,
            bo_x2_ref,
            sems,  # [4, 2]
        ) = refs
        lse_hbm_ref = None
        blse_x2_ref = None

    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    # Width of the stored output (q_nope's in split-q mode, else head_dim).
    o_dim = head_dim
    if split_q:
        head_dim = align_to(head_dim + q_pe_hbm_ref.shape[-1], 128)
    nope_dim = kv_cache_utils.WORD_BYTES * kv_cache_utils.TILE_LANE_BYTES
    assert kv_lens_ref.shape[0] == num_tokens
    bkv_sz = (
        cache_kv_nope_hbm_ref.shape[1]
        if sparsecore_layout
        else cache_kv_nope_hbm_ref.shape[1] // kv_cache_utils.WORD_BYTES
    )

    q_dtype = q_hbm_ref.dtype
    q_packing = get_dtype_packing(q_dtype)
    # Validate against the KV dtype.
    assert o_hbm_ref.dtype == q_dtype

    assert head_dim % 128 == 0
    assert num_q_heads % q_packing == 0

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def get_batch_max_kv_len(seq_idx_start):
        max_kv = jnp.int32(0)
        for b in range(batch_size):
            max_kv = jnp.maximum(max_kv, kv_lens_ref[seq_idx_start + b])
        return max_kv

    def attention_step1_qk_softmax(
        q,  # [num_q_heads, head_dim]
        kv,  # [slice_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
        kv_len,  # i32[] number of valid (non "-1") top-k rows in kv
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape[1] == head_dim

        # Single-block softmax: each query attends exactly one gathered kv
        # block, so no FlashAttention running-max/rescale state (GLM-5.2 DSA
        # has no SWA or sinks). The step1/step2 split only interleaves VPU
        # and MXU work across batch entries.
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        if k_scale != 1.0:
            s *= k_scale

        # Mask scores past kv_len (the "-1" tail): zeroing the keys alone is
        # not enough -- a zeroed key still scores 0 and inflates the softmax
        # denominator whenever kv_len << topk.
        kv_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(kv_span < kv_len, s, jnp.finfo(jnp.float32).min)

        m = jnp.max(s, axis=1, keepdims=True)
        p = jnp.exp(s - m)
        l_sum = jnp.sum(p, axis=1, keepdims=True)

        # `m` is needed only to reconstruct the log-sum-exp for a cross-shard
        # merge; the single-device path drops it.
        return p, l_sum, m

    def attention_step2_pv(
        p,
        kv,
        l_sum,
    ):
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        # We use `k_scale` for the PV step as well since MLA uses a shared
        # compressed latent for k/v and k_scale ~= v_scale.
        # TODO(b/539630772): Add a separate v_scale param and apply it to the
        # PV step.
        if k_scale != 1.0:
            pv *= k_scale

        out = (
            lax.div(pv, l_sum)
            if q_dtype == jnp.float32
            else (pv * pl.reciprocal(l_sum, approx=True)).astype(q_dtype)
        )
        return out

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    if is_dcp_mode:

        def _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, slice_sz, *, wait=False):
            sem_nope = sems.at[0, bkv_sem_idx]
            sem_rope = sems.at[3, bkv_sem_idx]
            nope_rows = (
                slice_sz
                if sparsecore_layout
                else slice_sz * kv_cache_utils.WORD_BYTES
            )
            rope_rows = (
                slice_sz // native_sc_gather.ROPE_PACKING
                if sparsecore_layout
                else slice_sz
            )
            bkv_nope_vmem_ref = bkv_nope_x2_ref.at[
                bkv_sem_idx, :, pl.ds(0, nope_rows), :
            ]
            bkv_rope_vmem_ref = bkv_rope_x2_ref.at[
                bkv_sem_idx, :, pl.ds(0, rope_rows), :
            ]
            page_idx_start = seq_idx_start - start_seq_idx
            if not wait:
                _async_copy(
                    cache_kv_nope_hbm_ref.at[
                        pl.ds(page_idx_start, batch_size), pl.ds(0, nope_rows), :
                    ],
                    bkv_nope_vmem_ref,
                    sem_nope,
                    wait,
                )
                _async_copy(
                    cache_kv_rope_hbm_ref.at[
                        pl.ds(page_idx_start, batch_size), pl.ds(0, rope_rows), :
                    ],
                    bkv_rope_vmem_ref,
                    sem_rope,
                    wait,
                )
            else:
                _async_copy(
                    src=bkv_nope_vmem_ref,
                    dst=bkv_nope_vmem_ref,
                    sem=sem_nope,
                    wait=True,
                )
                _async_copy(
                    src=bkv_rope_vmem_ref,
                    dst=bkv_rope_vmem_ref,
                    sem=sem_rope,
                    wait=True,
                )

        def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
            batch_max_kv = get_batch_max_kv_len(seq_idx_start)
            half_sz = bkv_sz // 2
            quarter_sz = bkv_sz // 4
            eighth_sz = bkv_sz // 8
            sixteenth_sz = bkv_sz // 16
            thirty_second_sz = bkv_sz // 32
            if (
                thirty_second_sz >= 64
                and thirty_second_sz % 64 == 0
                and bkv_sz % 32 == 0
            ):

                @pl.when((batch_max_kv > 0) & (batch_max_kv <= thirty_second_sz))
                def _fetch_thirty_second():
                    _fetch_bkv_slice(
                        seq_idx_start, bkv_sem_idx, thirty_second_sz, wait=wait
                    )

                @pl.when(
                    (batch_max_kv > thirty_second_sz) & (batch_max_kv <= sixteenth_sz)
                )
                def _fetch_sixteenth():
                    _fetch_bkv_slice(
                        seq_idx_start, bkv_sem_idx, sixteenth_sz, wait=wait
                    )

                @pl.when((batch_max_kv > sixteenth_sz) & (batch_max_kv <= eighth_sz))
                def _fetch_eighth():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, eighth_sz, wait=wait)

                @pl.when((batch_max_kv > eighth_sz) & (batch_max_kv <= quarter_sz))
                def _fetch_quarter():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, quarter_sz, wait=wait)

                @pl.when((batch_max_kv > quarter_sz) & (batch_max_kv <= half_sz))
                def _fetch_half():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, half_sz, wait=wait)

                @pl.when(batch_max_kv > half_sz)
                def _fetch_full():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, bkv_sz, wait=wait)
            elif (
                sixteenth_sz >= 64
                and sixteenth_sz % 64 == 0
                and bkv_sz % 16 == 0
            ):

                @pl.when((batch_max_kv > 0) & (batch_max_kv <= sixteenth_sz))
                def _fetch_sixteenth():
                    _fetch_bkv_slice(
                        seq_idx_start, bkv_sem_idx, sixteenth_sz, wait=wait
                    )

                @pl.when((batch_max_kv > sixteenth_sz) & (batch_max_kv <= eighth_sz))
                def _fetch_eighth():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, eighth_sz, wait=wait)

                @pl.when((batch_max_kv > eighth_sz) & (batch_max_kv <= quarter_sz))
                def _fetch_quarter():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, quarter_sz, wait=wait)

                @pl.when((batch_max_kv > quarter_sz) & (batch_max_kv <= half_sz))
                def _fetch_half():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, half_sz, wait=wait)

                @pl.when(batch_max_kv > half_sz)
                def _fetch_full():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, bkv_sz, wait=wait)
            elif eighth_sz >= 64 and eighth_sz % 64 == 0 and bkv_sz % 8 == 0:

                @pl.when((batch_max_kv > 0) & (batch_max_kv <= eighth_sz))
                def _fetch_eighth():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, eighth_sz, wait=wait)

                @pl.when((batch_max_kv > eighth_sz) & (batch_max_kv <= quarter_sz))
                def _fetch_quarter():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, quarter_sz, wait=wait)

                @pl.when((batch_max_kv > quarter_sz) & (batch_max_kv <= half_sz))
                def _fetch_half():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, half_sz, wait=wait)

                @pl.when(batch_max_kv > half_sz)
                def _fetch_full():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, bkv_sz, wait=wait)
            elif quarter_sz >= 64 and quarter_sz % 64 == 0 and bkv_sz % 4 == 0:

                @pl.when((batch_max_kv > 0) & (batch_max_kv <= quarter_sz))
                def _fetch_quarter():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, quarter_sz, wait=wait)

                @pl.when((batch_max_kv > quarter_sz) & (batch_max_kv <= half_sz))
                def _fetch_half():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, half_sz, wait=wait)

                @pl.when(batch_max_kv > half_sz)
                def _fetch_full():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, bkv_sz, wait=wait)
            elif half_sz >= 64 and half_sz % 64 == 0 and bkv_sz % 2 == 0:

                @pl.when((batch_max_kv > 0) & (batch_max_kv <= half_sz))
                def _fetch_half():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, half_sz, wait=wait)

                @pl.when(batch_max_kv > half_sz)
                def _fetch_full():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, bkv_sz, wait=wait)
            else:

                @pl.when(batch_max_kv > 0)
                def _fetch_full():
                    _fetch_bkv_slice(seq_idx_start, bkv_sem_idx, bkv_sz, wait=wait)
    else:

        def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
            sem_nope = sems.at[0, bkv_sem_idx]
            sem_rope = sems.at[3, bkv_sem_idx]

            bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
            bkv_rope_vmem_ref = bkv_rope_x2_ref.at[bkv_sem_idx, :]

            # The index into cache_kv_hbm_ref should be relative to the current
            # chunk.
            page_idx_start = seq_idx_start - start_seq_idx
            if not wait:
                _async_copy(
                    cache_kv_nope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                    bkv_nope_vmem_ref,
                    sem_nope,
                    wait,
                )
                _async_copy(
                    cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start, batch_size)],
                    bkv_rope_vmem_ref,
                    sem_rope,
                    wait,
                )
            else:
                dst_nope = bkv_nope_vmem_ref
                _async_copy(src=dst_nope, dst=dst_nope, sem=sem_nope, wait=True)
                dst_rope = bkv_rope_vmem_ref
                _async_copy(src=dst_rope, dst=dst_rope, sem=sem_rope, wait=True)

    def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        bq_vmem_ref = bq_x2_ref.at[bq_sem_idx, :]

        if not wait:
            _async_copy(
                q_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                bq_vmem_ref,
                sem,
                wait,
            )
            if split_q:
                # Same semaphore; the waits below mirror both starts.
                _async_copy(
                    q_pe_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                    bq_pe_x2_ref.at[bq_sem_idx, :],
                    sem,
                    wait,
                )
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)
            if split_q:
                bq_pe_vmem_ref = bq_pe_x2_ref.at[bq_sem_idx, :]
                _async_copy(
                    src=bq_pe_vmem_ref, dst=bq_pe_vmem_ref, sem=sem, wait=True
                )

    def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx, :]

        if not wait:
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _send_blse_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[4, bo_sem_idx]
        vmem_ref = blse_x2_ref.at[bo_sem_idx, :]

        if not wait:
            _async_copy(
                vmem_ref,
                lse_hbm_ref.at[pl.ds(seq_idx_start, batch_size)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx)

    def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True)

    def start_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx)

    def wait_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx, wait=True)

    def load_bq(bq_sem_idx, batch_idx):
        q = bq_x2_ref.at[bq_sem_idx, batch_idx][...]
        if split_q:
            # The caller's concat + zero pad, rebuilt in VMEM.
            q = jnp.concatenate([q, bq_pe_x2_ref.at[bq_sem_idx, batch_idx][...]], -1)
            if q.shape[-1] < head_dim:
                q = jnp.pad(q, ((0, 0), (0, head_dim - q.shape[-1])))
        return q

    def store_blse(bo_sem_idx, batch_idx, lse_col):
        """Stores one token's f32[num_q_heads, 1] log-sum-exp column."""
        if not lse_lane_dense:
            blse_x2_ref.at[bo_sem_idx, batch_idx][...] = lse_col
            return
        # Heads from sublanes to lanes, exactly: each column of the masked
        # [heads, heads] tile holds one live entry, the rest -inf.
        eye_shape = (num_q_heads, num_q_heads)
        eye = lax.broadcasted_iota(jnp.int32, eye_shape, 0) == (
            lax.broadcasted_iota(jnp.int32, eye_shape, 1)
        )
        row = jnp.max(
            jnp.where(eye, jnp.broadcast_to(lse_col, eye_shape), -jnp.inf),
            axis=0,
            keepdims=True,
        )
        tile_ref = blse_x2_ref.at[bo_sem_idx]  # [bsz, num_q_heads]
        if isinstance(batch_idx, int):
            tile_ref[pl.ds(batch_idx, 1), :] = row
        else:
            pick = lax.broadcasted_iota(jnp.int32, tile_ref.shape, 0) == batch_idx
            tile_ref[...] = jnp.where(pick, row, tile_ref[...])

    def load_bkv(bkv_sem_idx, batch_idx, slice_sz=bkv_sz):
        if is_dcp_mode:
            if sparsecore_layout:
                bkv_nope = bkv_nope_x2_ref.at[
                    bkv_sem_idx, batch_idx, pl.ds(0, slice_sz), :
                ][...]
                bkv_nope = pltpu.bitcast(bkv_nope, jnp.uint8)
            else:
                bkv_nope = bkv_nope_x2_ref.at[
                    bkv_sem_idx,
                    batch_idx,
                    pl.ds(0, slice_sz * kv_cache_utils.WORD_BYTES),
                    :,
                ][...]
            bkv_nope = bkv_nope.reshape(slice_sz, nope_dim)
            bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

            # Same for rope, except four tokens share a row there.
            if sparsecore_layout:
                bkv_rope = bkv_rope_x2_ref.at[
                    bkv_sem_idx,
                    batch_idx,
                    pl.ds(0, slice_sz // native_sc_gather.ROPE_PACKING),
                    :,
                ][...]
                bkv_rope = pltpu.bitcast(bkv_rope, jnp.uint8)
                bkv_rope = bkv_rope.reshape(slice_sz, 128)
            else:
                bkv_rope = bkv_rope_x2_ref.at[
                    bkv_sem_idx, batch_idx, pl.ds(0, slice_sz), :
                ][...]
            bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)
            bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

            # Pad kv (576) to q's 128-aligned head_dim (640); the padded dims are
            # no-ops in the einsums since q's padded dims are also zero.
            if bkv.shape[-1] < head_dim:
                bkv = jnp.pad(bkv, ((0, 0), (0, head_dim - bkv.shape[-1])))

            # Zero keys beyond kv_len: the "-1" tail is gathered from scattered
            # pages and can decode to fp8 NaN; score masking zeroes the weights,
            # but 0 * NaN = NaN in the PV einsum, so the data must be finite too.
            kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
            k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
            bkv = jnp.where(k_span < kv_len, bkv, 0)
            return bkv
        else:
            bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
            if sparsecore_layout:
                # Native SC returns one uint32[128] row per token. Bitcasting
                # expands the packed byte lanes before folding them into the
                # logical 512-byte NOPE vector.
                bkv_nope = pltpu.bitcast(bkv_nope, jnp.uint8)
            bkv_nope = bkv_nope.reshape(bkv_sz, nope_dim)
            bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

            bkv_rope = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...]
            if sparsecore_layout:
                # The native gather's delta-swap produces four-token grouped
                # uint32[128] rows. Expanding their byte lanes recovers one
                # contiguous 128-byte row per selected token.
                bkv_rope = pltpu.bitcast(bkv_rope, jnp.uint8)
                bkv_rope = bkv_rope.reshape(bkv_sz, 128)
            bkv_rope = pltpu.bitcast(bkv_rope, jnp.float8_e4m3fn)
            bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)

            if bkv.shape[-1] < head_dim:
                bkv = jnp.pad(bkv, ((0, 0), (0, head_dim - bkv.shape[-1])))

            # Zero keys beyond kv_len: the tail decodes to fp8 NaN and
            # 0 * NaN = NaN in the PV einsum, so masking the scores is not enough.
            kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
            k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
            bkv = jnp.where(k_span < kv_len, bkv, 0)
            return bkv

    def process():
        def get_next_seq_ids(seq_idx, bi_sem_idx):
            next_seq_idx = seq_idx + batch_size
            next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bi_sem_idx

        bi_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bi_sem_idx = get_next_seq_ids(
            batch_start_seq_idx, bi_sem_idx
        )

        # Prefetch next seq
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_seq():
            sem_ids_ref[0] = next_bi_sem_idx
            start_fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

        def write_blse(batch_idx, m, l_sum):
            # log-sum-exp in the *unshifted* frame, so a peer holding a
            # disjoint slice of the same token's top-k can merge with
            # o = sum_r softmax(lse_r) * o_r without knowing our max.
            if not return_lse:
                return
            if is_dcp_mode:
                kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
                store_blse(
                    bo_sem_idx,
                    batch_idx,
                    jnp.where(kv_len > 0, m + jnp.log(l_sum), -jnp.inf),
                )
            else:
                store_blse(bo_sem_idx, batch_idx, m + jnp.log(l_sum))

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(batch_start_seq_idx, bo_sem_idx)
            if return_lse:
                _send_blse_batch(batch_start_seq_idx, bo_sem_idx, wait=True)

        if is_dcp_mode:

            def _run_batch_loop(slice_sz: int):
                prev_p = None
                prev_bkv = None
                prev_l_sum = None
                prev_m = None
                prev_out = None

                for batch_idx in range(batch_size):
                    kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
                    bkv = load_bkv(bi_sem_idx, batch_idx, slice_sz=slice_sz)
                    bq = load_bq(bi_sem_idx, batch_idx)

                    if prev_out is not None:
                        # Artificial dependency to force MXU/VPU interleaving by limiting
                        # LLO's QK runahead
                        # We use jnp.where to prevent XLA from optimizing the dependency away.
                        # prev_out won't be inf in practice.
                        bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

                    p, l_sum, m = attention_step1_qk_softmax(bq, bkv, kv_len)

                    if prev_p is not None:
                        assert prev_bkv is not None
                        assert prev_l_sum is not None
                        out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)

                        # Store output from acc to bo.
                        bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out[:, :o_dim]
                        write_blse(batch_idx - 1, prev_m, prev_l_sum)
                        prev_out = out

                    prev_p = p
                    prev_bkv = bkv
                    prev_l_sum = l_sum
                    prev_m = m

                # end of pipelining loop
                out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)
                bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out[:, :o_dim]
                write_blse(batch_size - 1, prev_m, prev_l_sum)

            batch_max_kv_len = get_batch_max_kv_len(batch_start_seq_idx)

            @pl.when(batch_max_kv_len == 0)
            def _zero_batch():
                zero_out = jnp.zeros((num_q_heads, o_dim), dtype=q_dtype)
                neg_inf_lse = jnp.full((num_q_heads, 1), -jnp.inf, dtype=jnp.float32)
                for batch_idx in range(batch_size):
                    bo_x2_ref.at[bo_sem_idx, batch_idx][...] = zero_out
                    if return_lse:
                        store_blse(bo_sem_idx, batch_idx, neg_inf_lse)

            thirty_second_sz = bkv_sz // 32
            sixteenth_sz = bkv_sz // 16
            eighth_sz = bkv_sz // 8
            quarter_sz = bkv_sz // 4
            qtr_sz = quarter_sz
            half_sz = bkv_sz // 2
            if (
                thirty_second_sz >= 64
                and thirty_second_sz % 64 == 0
                and bkv_sz % 32 == 0
            ):

                @pl.when(
                    (batch_max_kv_len > 0) & (batch_max_kv_len <= thirty_second_sz)
                )
                def _run_thirty_second():
                    _run_batch_loop(thirty_second_sz)

                @pl.when(
                    (batch_max_kv_len > thirty_second_sz)
                    & (batch_max_kv_len <= sixteenth_sz)
                )
                def _run_sixteenth():
                    _run_batch_loop(sixteenth_sz)

                @pl.when(
                    (batch_max_kv_len > sixteenth_sz) & (batch_max_kv_len <= eighth_sz)
                )
                def _run_eighth():
                    _run_batch_loop(eighth_sz)

                @pl.when((batch_max_kv_len > eighth_sz) & (batch_max_kv_len <= qtr_sz))
                def _run_qtr():
                    _run_batch_loop(qtr_sz)

                @pl.when((batch_max_kv_len > qtr_sz) & (batch_max_kv_len <= half_sz))
                def _run_half():
                    _run_batch_loop(half_sz)

                @pl.when(batch_max_kv_len > half_sz)
                def _run_full():
                    _run_batch_loop(bkv_sz)
            elif (
                sixteenth_sz >= 64
                and sixteenth_sz % 64 == 0
                and bkv_sz % 16 == 0
            ):

                @pl.when(
                    (batch_max_kv_len > 0) & (batch_max_kv_len <= sixteenth_sz)
                )
                def _run_sixteenth():
                    _run_batch_loop(sixteenth_sz)

                @pl.when(
                    (batch_max_kv_len > sixteenth_sz) & (batch_max_kv_len <= eighth_sz)
                )
                def _run_eighth():
                    _run_batch_loop(eighth_sz)

                @pl.when((batch_max_kv_len > eighth_sz) & (batch_max_kv_len <= qtr_sz))
                def _run_qtr():
                    _run_batch_loop(qtr_sz)

                @pl.when((batch_max_kv_len > qtr_sz) & (batch_max_kv_len <= half_sz))
                def _run_half():
                    _run_batch_loop(half_sz)

                @pl.when(batch_max_kv_len > half_sz)
                def _run_full():
                    _run_batch_loop(bkv_sz)
            elif eighth_sz >= 64 and eighth_sz % 64 == 0 and bkv_sz % 8 == 0:

                @pl.when((batch_max_kv_len > 0) & (batch_max_kv_len <= eighth_sz))
                def _run_eighth():
                    _run_batch_loop(eighth_sz)

                @pl.when((batch_max_kv_len > eighth_sz) & (batch_max_kv_len <= qtr_sz))
                def _run_qtr():
                    _run_batch_loop(qtr_sz)

                @pl.when((batch_max_kv_len > qtr_sz) & (batch_max_kv_len <= half_sz))
                def _run_half():
                    _run_batch_loop(half_sz)

                @pl.when(batch_max_kv_len > half_sz)
                def _run_full():
                    _run_batch_loop(bkv_sz)
            elif qtr_sz >= 64 and qtr_sz % 64 == 0 and bkv_sz % 4 == 0:

                @pl.when((batch_max_kv_len > 0) & (batch_max_kv_len <= qtr_sz))
                def _run_qtr():
                    _run_batch_loop(qtr_sz)

                @pl.when((batch_max_kv_len > qtr_sz) & (batch_max_kv_len <= half_sz))
                def _run_half():
                    _run_batch_loop(half_sz)

                @pl.when(batch_max_kv_len > half_sz)
                def _run_full():
                    _run_batch_loop(bkv_sz)
            elif half_sz >= 64 and half_sz % 64 == 0 and bkv_sz % 2 == 0:

                @pl.when((batch_max_kv_len > 0) & (batch_max_kv_len <= half_sz))
                def _run_half():
                    _run_batch_loop(half_sz)

                @pl.when(batch_max_kv_len > half_sz)
                def _run_full():
                    _run_batch_loop(bkv_sz)
            else:

                @pl.when(batch_max_kv_len > 0)
                def _run_full():
                    _run_batch_loop(bkv_sz)
        else:
            prev_p = None
            prev_bkv = None
            prev_l_sum = None
            prev_m = None
            prev_out = None

            for batch_idx in range(batch_size):
                kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
                bkv = load_bkv(bi_sem_idx, batch_idx)
                bq = load_bq(bi_sem_idx, batch_idx)

                if prev_out is not None:
                    # Artificial dependency to force MXU/VPU interleaving by limiting
                    # LLO's QK runahead
                    # We use jnp.where to prevent XLA from optimizing the dependency away.
                    # prev_out won't be inf in practice.
                    bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

                p, l_sum, m = attention_step1_qk_softmax(bq, bkv, kv_len)

                if prev_p is not None:
                    assert prev_bkv is not None
                    assert prev_l_sum is not None
                    out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)

                    # Store output from acc to bo.
                    bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out[:, :o_dim]
                    write_blse(batch_idx - 1, prev_m, prev_l_sum)
                    prev_out = out

                prev_p = p
                prev_bkv = bkv
                prev_l_sum = l_sum
                prev_m = m

            # end of pipelining loop
            out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)
            bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out[:, :o_dim]
            write_blse(batch_size - 1, prev_m, prev_l_sum)
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)
        if return_lse:
            _send_blse_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    # b/567663022: the grid is static (chunk_size // batch_size) so it does not
    # depend on cu_q_lens/distribution; only the first `num_active` programs
    # (those covering [start_seq_idx, end_seq_idx)) do any work. The caller
    # guarantees (end_seq_idx - start_seq_idx) % batch_size == 0 and
    # end_seq_idx >= start_seq_idx. Under a dynamic grid num_active equals
    # pl.num_programs(0), so this is a no-op there.
    num_active = (end_seq_idx - start_seq_idx) // batch_size

    @pl.when(pl.program_id(0) < num_active)
    def _active():
        @pl.when(batch_start_seq_idx == start_seq_idx)
        def prologue():
            start_fetch_bq_batch(batch_start_seq_idx, 0)
            start_fetch_bkv_batch(batch_start_seq_idx, 0)

        process()

        @pl.when(batch_end_seq_idx == end_seq_idx - 1)
        def epilogue():
            # The first argument "0" for seq_idx_start does not matter here.
            wait_send_bo_batch(0, 0)
            if return_lse:
                _send_blse_batch(0, 0, wait=True)

            # Must key off the ACTIVE program count, not pl.num_programs(0):
            # with a static grid and one active program, slot 1 never had a
            # send started, so waiting on it would hang.
            @pl.when(num_active >= 2)
            def _wait_1():
                # The first argument "0" for seq_idx_start does not matter.
                wait_send_bo_batch(0, 1)
                if return_lse:
                    _send_blse_batch(0, 1, wait=True)

    ### ------- Kernel end ------- ###


def prepare_q_inputs(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, actual_head_dim],
):
    _, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q,
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    return q


def prepare_outputs(
    out,  # [max_num_tokens, num_q_heads, head_dim]
    actual_num_q_heads: int,
    actual_head_dim: int,
):
    return out[:, :actual_num_q_heads, :actual_head_dim]


# Main attention kernel for GLM-5.2 DSA (top-k gather and attention).
# The KV entries referenced by `topk_indices` (including the current batch's
# own tokens) must already be resident in the caches before this call;
# per-token kv_lens derive from the "-1" padding in `topk_indices`.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "k_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
        "return_lse",
        "cache_layout",
        "sparsecore_atoms_per_batch",
        "sparsecore_gather_wait_mode",
        "sparsecore_gather_pipelined",
        "is_dcp_mode",
        "lse_lane_dense",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    # TensorCore: u8[pages, page_size, 4, 128].
    # SparseCore: u32[pages, page_size, 128].
    cache_kv_nope: jax.Array,
    # TensorCore: u8[pages, page_size/4, 4, 128].
    # SparseCore: u32[pages, page_size/4, 128].
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array | None,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    k_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = 64,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    return_lse: bool = False,
    cache_layout: str = TENSORCORE_CACHE_LAYOUT,
    sparsecore_atoms_per_batch: int = 16,
    sparsecore_gather_wait_mode: str = "overlap_rope",
    sparsecore_gather_pipelined: bool = True,
    is_dcp_mode: bool = False,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """MLA Ragged paged attention that supports mixed prefill and decode.

    Args:
      q: concatenated all sequences' queries.
      cache_kv_nope: the current kv cache for nope.
      cache_kv_rope: the current kv cache for rope.
      topk_indices: for each query token, the indices of the top k key tokens to
        attend to.
      page_indices: flattened page indices look-up table by (seq_id, page_id).
      cu_q_lens: the cumulative sum of the effective query lengths. Similar to
        kv_lens, only the first num_seqs+1 values are valid.
      distribution: (i, j, k) represents that sequences[0:i] are decode-only,
        sequences[i:j] are chunked-prefill-only, and sequences[j:k] are mixed. The
        k is also the total number of sequences.
      sm_scale: the softmax scale which will be applied to the Q@K^T.
      k_scale: per-tensor dequantization scale for the fp8 nope cache.
      vmem_limit_bytes: the vmem limit for the pallas kernel.
      return_lse: also return the per-(token, head) log-sum-exp of the attention
        scores. Needed only when `topk_indices` holds a slice of a token's
        global top-k and the partial outputs of several ranks must be merged.
      cache_layout: Physical cache layout: ``"tensorcore"`` for tiled uint8
        caches or ``"sparsecore"`` for native uint32 caches. Both modes perform
        the same sparse attention computation.
      sparsecore_atoms_per_batch: Native SparseCore gather pipeline atoms per
        owner batch.
      sparsecore_gather_wait_mode: Whether native SparseCore NOPE output DMA
        overlaps ROPE work.
      sparsecore_gather_pipelined: Whether to pipeline native SparseCore gather
        serving.
      is_dcp_mode: whether to enable DCP mode (skips gather and attention compute
        for padding and uses dynamic KV cache slice sizing).

    Returns:
      The output of attention, or `(output, lse)` when `return_lse` is set.
      `lse` is f32[max_num_tokens, actual_num_q_heads].
    """
    if cache_layout not in (TENSORCORE_CACHE_LAYOUT, SPARSECORE_CACHE_LAYOUT):
        raise ValueError(
            f"cache_layout must be {TENSORCORE_CACHE_LAYOUT!r} or "
            f"{SPARSECORE_CACHE_LAYOUT!r}, got {cache_layout!r}"
        )
    sparsecore_layout = cache_layout == SPARSECORE_CACHE_LAYOUT
    if sparsecore_layout:
        if (
            cache_kv_nope.dtype != jnp.uint32
            or cache_kv_nope.ndim != 3
            or cache_kv_nope.shape[-1] != 128
        ):
            raise ValueError(
                "sparsecore NOPE cache must be uint32[pages,page_size,128]"
            )
        if (
            cache_kv_rope.dtype != jnp.uint32
            or cache_kv_rope.ndim != 3
            or cache_kv_rope.shape[-1] != 128
        ):
            raise ValueError(
                "sparsecore ROPE cache must be uint32[pages,page_size/4,128]"
            )
    else:
        if (
            cache_kv_nope.dtype != jnp.uint8
            or cache_kv_nope.ndim != 4
            or cache_kv_nope.shape[-2:] != (4, 128)
        ):
            raise ValueError(
                "tensorcore NOPE cache must be uint8[pages,page_size,4,128]"
            )
        if (
            cache_kv_rope.dtype != jnp.uint8
            or cache_kv_rope.ndim != 4
            or cache_kv_rope.shape[-2:] != (4, 128)
        ):
            raise ValueError(
                "tensorcore ROPE cache must be uint8[pages,page_size/4,4,128]"
            )
    if cache_kv_nope.size != 4 * cache_kv_rope.size:
        raise ValueError(
            f"NOPE cache size ({cache_kv_nope.size}) must be exactly "
            f"4 * ROPE cache size ({cache_kv_rope.size})"
        )
    if (
        cache_kv_nope.shape[0] != cache_kv_rope.shape[0]
        or cache_kv_nope.shape[1] != 4 * cache_kv_rope.shape[1]
    ):
        raise ValueError(
            "NOPE and ROPE caches must have matching page counts and page geometry"
        )

    _, actual_num_q_heads, actual_head_dim = q.shape

    if q_pe is not None:
        if q_pe.shape[:2] != q.shape[:2] or q_pe.dtype != q.dtype:
            raise ValueError(
                f"q_pe {q_pe.shape}/{q_pe.dtype} must match q_nope "
                f"{q.shape}/{q.dtype} in tokens, heads and dtype"
            )
        if actual_head_dim % 128 or actual_num_q_heads % get_dtype_packing(q.dtype):
            raise ValueError(
                "split q needs a 128-multiple nope dim and packed heads, got "
                f"{q.shape}"
            )
    else:
        q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    head_dim = q.shape[-1]

    page_size = cache_kv_nope.shape[1]

    _, num_q_heads, _ = q.shape
    max_num_seqs = cu_q_lens.shape[0] - 1

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        # [num_tokens, topk * WORD_BYTES, TILE_LANE_BYTES] (gathered raw tiles)
        cache_kv_nope: jax.Array,
        cache_kv_rope: jax.Array,  # [num_tokens, topk, rope_dim] (gathered)
        kv_lens: jax.Array,  # i32[max_num_tokens]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        kernel_batch_size: int,
        num_batches: int,  # static grid size (upper bound on active batches)
        lse: jax.Array | None = None,  # f32[max_num_tokens, num_q_heads, 1] or f32[max_num_tokens, num_q_heads]
        is_dcp_mode: bool = False,
        q_pe: jax.Array | None = None,
        lse_lane_dense: bool = False,
    ):
        batch_size = kernel_batch_size
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        # b/567663022: static grid; the kernel skips programs at/after the
        # dynamic end_seq_idx via its `num_active` guard.
        grid = (num_batches,)
        split_q = q_pe is not None
        scalar_prefetches = (
            kv_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2,), jnp.int32),
        )
        num_prefetch = len(scalar_prefetches)
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        ]
        if split_q:
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o
        # Each chunk writes only its own token rows, so -- exactly like `q` --
        # the lse buffer is threaded through as an aliased in/out to carry
        # earlier chunks' rows forward.
        if return_lse:
            in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))  # lse (in)
            out_specs = [out_specs, pl.BlockSpec(memory_space=pltpu.HBM)]

        # One batch entry's worth of gathered top-k rows, per cache.
        bkv_nope_double_buf = pltpu.VMEM(
            (2, batch_size, *cache_kv_nope.shape[1:]),
            cache_kv_nope.dtype,
        )
        bkv_rope_double_buf = pltpu.VMEM(
            (2, batch_size, *cache_kv_rope.shape[1:]),
            cache_kv_rope.dtype,
        )

        bq_double_buf = pltpu.VMEM(
            (2, batch_size, num_q_heads, q.shape[-1]),
            q.dtype,
        )

        bo_double_buf = bq_double_buf

        scratch_shapes = [
            bkv_nope_double_buf,
            bkv_rope_double_buf,
            bq_double_buf,
            bo_double_buf,  # Double buffering for output block.
        ]
        if return_lse:
            scratch_shapes.append(
                pltpu.VMEM(
                    (2, batch_size, num_q_heads)
                    if lse_lane_dense
                    else (2, batch_size, num_q_heads, 1),
                    jnp.float32,
                )
            )
        # Semaphores for double buffering of bkv_nope, bq, bo, bkv_rope
        # (+ blse when the log-sum-exp is exported).
        scratch_shapes.append(pltpu.SemaphoreType.DMA((5 if return_lse else 4, 2)))
        if split_q:
            scratch_shapes.append(
                pltpu.VMEM((2, batch_size, num_q_heads, q_pe.shape[-1]), q_pe.dtype)
            )

        scope_name = (
            f"MLA-{cache_layout}-p_{cache_kv_rope.shape[1]}-"
            f"bz_{batch_size}-gcz_{cache_kv_nope.shape[0]}"
        )
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel,
                    sm_scale=sm_scale,
                    k_scale=k_scale,
                    batch_size=batch_size,
                    return_lse=return_lse,
                    sparsecore_layout=sparsecore_layout,
                    is_dcp_mode=is_dcp_mode,
                    split_q=split_q,
                    lse_lane_dense=lse_lane_dense,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=(
                    [
                        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                        jax.ShapeDtypeStruct(shape=lse.shape, dtype=lse.dtype),
                    ]
                    if return_lse
                    else jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
                ),
                input_output_aliases=(
                    {
                        num_prefetch: 0,
                        num_prefetch + 3 + int(split_q): 1,
                    }
                    if return_lse
                    else {
                        num_prefetch: 0,  # Alias output activation with q
                    }
                ),
                name=scope_name,
            )
        )
        operands = [q, cache_kv_nope, cache_kv_rope]
        if split_q:
            operands.append(q_pe)
        if return_lse:
            operands.append(lse)
        return kernel(*scalar_prefetches, *operands)

    if topk_indices.dtype != jnp.int32 or topk_indices.ndim != 2:
        raise ValueError("topk_indices must be int32[num_tokens, topk]")
    if q.shape[0] != topk_indices.shape[0]:
        raise ValueError(
            f"q tokens ({q.shape[0]}) must match topk_indices tokens "
            f"({topk_indices.shape[0]})"
        )
    topk = topk_indices.shape[-1]
    if sparsecore_layout and topk % native_sc_gather.ROPE_PACKING != 0:
        raise ValueError(
            f"topk ({topk}) must be a multiple of "
            f"{native_sc_gather.ROPE_PACKING} for sparsecore ROPE packing"
        )

    num_tokens = q.shape[0]
    num_valid_tokens = cu_q_lens[distribution[2]]
    is_padding = topk_indices == -1
    flat_element_index = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
        num_tokens, topk
    )
    if page_indices is not None:
        num_page_indices = page_indices.shape[0]
        if num_page_indices % max_num_seqs != 0:
            raise ValueError(
                f"page_indices size ({num_page_indices}) must be divisible by "
                f"the sequence count ({max_num_seqs})"
            )
        tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
        seq_ids_segment = jnp.repeat(
            jnp.arange(max_num_seqs), tokens_per_seq, total_repeat_length=num_tokens
        )
        seq_page_ids = topk_indices // page_size
        token_offset = topk_indices % page_size
        page_ids = gather_page_ids(
            page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
        )

        # Scatter "-1" padding across the cache to avoid read hot spots.
        total_num_pages = cache_kv_nope.shape[0]
        scattered_page_ids = (flat_element_index * 104729) % total_num_pages
        scattered_token_offset = (flat_element_index * 15485863) % page_size
        page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
        token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
        physical_indices = page_ids * page_size + token_offset
    else:
        total_tokens = cache_kv_nope.shape[0] * page_size
        scattered_physical = (flat_element_index * 15485863) % total_tokens
        physical_indices = jnp.where(is_padding, scattered_physical, topk_indices)

    physical_indices = physical_indices.astype(jnp.int32)

    if is_dcp_mode:
        kv_lens = jnp.sum(topk_indices != -1, axis=-1)
        kv_lens = jnp.where(
            jnp.arange(num_tokens, dtype=jnp.int32) < num_valid_tokens,
            kv_lens,
            0,
        )
        gather_step_size = native_sc_gather.OCCURRENCES_PER_ATOM
        if topk >= gather_step_size and topk % gather_step_size == 0:
            tc_steps_per_window = _largest_divisor(topk // gather_step_size, 4)
            tc_num_row_subchunks = tc_steps_per_window * 4
            sc_atoms_per_batch = _largest_divisor(
                topk // gather_step_size, sparsecore_atoms_per_batch
            )
            steps_per_window = (
                sc_atoms_per_batch if sparsecore_layout else tc_steps_per_window
            )
            gather_window_size = steps_per_window * gather_step_size
            cols = jnp.arange(topk, dtype=jnp.int32)
            window_start = (cols // gather_window_size) * gather_window_size
            in_win_step = (cols - window_start) // gather_step_size
            step_rungs: list[int] = []
            _r = 1
            while _r < steps_per_window:
                step_rungs.append(_r)
                _r *= 2
            step_rungs.append(steps_per_window)
            prev_steps = jnp.zeros_like(in_win_step)
            for _idx in range(1, len(step_rungs)):
                _lo = step_rungs[_idx - 1]
                _hi = step_rungs[_idx]
                prev_steps = jnp.where(
                    (in_win_step >= _lo) & (in_win_step < _hi),
                    _lo,
                    prev_steps,
                )
            rung_triggers = window_start + prev_steps * gather_step_size
            physical_indices = jnp.where(
                rung_triggers[None, :] >= kv_lens[:, None],
                dsa_gather.SKIP_INDEX,
                physical_indices,
            )
        else:
            tc_num_row_subchunks = 32
            sc_atoms_per_batch = sparsecore_atoms_per_batch
    else:
        kv_lens = jnp.sum(topk_indices != -1, axis=-1)
        tc_num_row_subchunks = 32
        sc_atoms_per_batch = sparsecore_atoms_per_batch

    num_chunks = cdiv(num_tokens, gather_and_attention_chunk_size)
    _log_patch_once(
        "Applied TPU patch: b/567663022 sparse-MLA static grid + 1-ahead "
        "gather pipelining (layout=%s, %d tokens -> %d chunk(s) of %d, dcp_mode=%s)%s.",
        cache_layout,
        num_tokens,
        num_chunks,
        gather_and_attention_chunk_size,
        is_dcp_mode,
        "; one chunk, so nothing to pipeline" if num_chunks == 1 else "",
    )

    # Rows the grid never reaches (padding past `batch_end`) keep this
    # sentinel, so a downstream merge weights them at exp(min - max) == 0
    # instead of tripping on uninitialised memory.
    lse = (
        jnp.full(
            (num_tokens, num_q_heads)
            if lse_lane_dense
            else (num_tokens, num_q_heads, 1),
            jnp.finfo(jnp.float32).min,
            dtype=jnp.float32,
        )
        if return_lse
        else None
    )

    def gather_chunk(i):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = min(start_pos + gather_and_attention_chunk_size, num_tokens)
        chunk_size = end_pos - start_pos
        indices = physical_indices[start_pos:end_pos]

        # Batching
        kernel_batch_size = _largest_divisor(chunk_size, attention_kernel_batch_size)
        assert chunk_size % kernel_batch_size == 0
        # The kernel grid walks [start_pos, batch_end) in `kernel_batch_size`
        # steps, so `batch_end - start_pos` MUST be a multiple of
        # `kernel_batch_size`.
        batch_end = start_pos + (
            cdiv(
                jnp.maximum(
                    0,
                    jnp.minimum(
                        num_valid_tokens,
                        end_pos,
                    )
                    - start_pos,
                ),
                kernel_batch_size,
            )
            * kernel_batch_size
        )
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

        # For prefilling of short sequences (or early in the sequence), there are
        # very few number of KVs in the sequence, so different qs' selected topk
        # would have large overlap. This causes gather read hotspotting. We've seen
        # 30%+ performance degradation compared to the no-duplicate-indices case.
        #
        # TODO: we could consider let the caller (tpu-runner) to sort the sequences
        # based on their lengths. For the sequences-segment below certain length,
        # we use a different kernel (dense attention and mask), for the rest of
        # sequences, we use this gather-and-attention kernel.
        if sparsecore_layout:
            gathered_nope_buffer, gathered_rope_buffer = (
                native_sc_gather.dsa_gather_native_sc(
                    cache_kv_nope,
                    cache_kv_rope,
                    indices,
                    out_size=chunk_size * topk,
                    atoms_per_batch=sc_atoms_per_batch,
                    wait_mode=sparsecore_gather_wait_mode,
                    serve_pipelined=sparsecore_gather_pipelined,
                    is_dcp_mode=is_dcp_mode,
                )
            )
            gathered_nope_buffer = gathered_nope_buffer.reshape(chunk_size, topk, 128)
            gathered_rope_buffer = gathered_rope_buffer.reshape(
                chunk_size, topk // native_sc_gather.ROPE_PACKING, 128
            )
        else:
            gathered_nope_buffer, gathered_rope_buffer = dsa_gather.dsa_gather(
                cache_kv_nope,
                cache_kv_rope,
                indices.reshape(-1),
                num_valid_indices=num_valid_indices,
                num_row_subchunks=tc_num_row_subchunks,
                is_dcp_mode=is_dcp_mode,
            )
            gathered_nope_buffer = gathered_nope_buffer.reshape(
                chunk_size,
                topk * kv_cache_utils.WORD_BYTES,
                kv_cache_utils.TILE_LANE_BYTES,
            )
            gathered_rope_buffer = gathered_rope_buffer.reshape(chunk_size, topk, -1)
        return (
            start_pos,
            chunk_size,
            kernel_batch_size,
            batch_end,
            gathered_nope_buffer,
            gathered_rope_buffer,
        )

    # b/567663022: 1-ahead software pipelining. The SparseCore gather for chunk
    # i+1 is emitted BEFORE the TensorCore MLA kernel for chunk i, so XLA has
    # an independent gather available to overlap with MLA(i) (MLA(i) itself
    # still depends on gather(i)). Numerically identical to the sequential
    # loop: the gather only reads the caches, which this loop never writes.
    # Costs one extra chunk of gathered KV live at a time.
    next_chunk = gather_chunk(0)
    for i in range(num_chunks):
        (
            start_pos,
            chunk_size,
            kernel_batch_size,
            batch_end,
            gathered_nope_buffer,
            gathered_rope_buffer,
        ) = next_chunk
        if i + 1 < num_chunks:
            next_chunk = gather_chunk(i + 1)
        # We treat each query token as a one independent sequence, attend to their
        # respective gathered kv tokens in the `gathered_kv_buffer`.
        # -1 in topk_indices is padded elements at the end of each row.
        result = run_mla_kernel(
            q,
            gathered_nope_buffer,
            gathered_rope_buffer,
            kv_lens,
            start_seq_idx=start_pos,
            end_seq_idx=batch_end,
            kernel_batch_size=kernel_batch_size,
            num_batches=chunk_size // kernel_batch_size,
            lse=lse,
            is_dcp_mode=is_dcp_mode,
            q_pe=q_pe,
            lse_lane_dense=lse_lane_dense,
        )
        if return_lse:
            q, lse = result
        else:
            q = result

    # [max_num_tokens, actual_num_q_heads, actual_head_dim]
    out = prepare_outputs(q, actual_num_q_heads, actual_head_dim)
    if return_lse:
        if lse_lane_dense:
            return out, lse[:, :actual_num_q_heads]
        return out, lse[:, :actual_num_q_heads, 0]
    return out


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "k_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "vmem_limit_bytes",
        "cache_layout",
        "sparsecore_atoms_per_batch",
        "sparsecore_gather_wait_mode",
        "sparsecore_gather_pipelined",
        "lse_lane_dense",
    ),
)
def dispatch_sparse_mla_6rung(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array | None,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    k_scale: float = 1.0,
    gather_and_attention_chunk_size: int = 64,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    cache_layout: str = TENSORCORE_CACHE_LAYOUT,
    sparsecore_atoms_per_batch: int = 16,
    sparsecore_gather_wait_mode: str = "overlap_rope",
    sparsecore_gather_pipelined: bool = True,
    q_pe: jax.Array | None = None,
    lse_lane_dense: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Lossless 6-Rung Dynamic KV Slice Dispatch with 0-Bypass Fastpath.

    Selects the active KV slice rung from `_RAGGED_K_RUNGS = (64, 128, 256, 512, 1024, 2048)`
    via `jax.lax.switch` based on `max_k = jnp.max(jnp.sum(topk_indices >= 0, axis=-1))`.
    Branch 0 (max_k == 0) returns zeros and -inf LSE immediately without kernel launch.
    Branches 1..6 slice `topk_indices` to the exact rung width, maintaining 16-row
    SparseCore ROPE DMA alignment and eliminating up to 32x VMEM and gather overhead.
    """
    actual_k = jnp.sum(topk_indices >= 0, axis=-1)
    max_k = jnp.max(actual_k)

    branch_idx = jnp.int32(0)
    for idx, rung in enumerate(_RAGGED_K_RUNGS, start=1):
        prev_rung = _RAGGED_K_RUNGS[idx - 2] if idx > 1 else 0
        branch_idx = jnp.where(max_k > prev_rung, jnp.int32(idx), branch_idx)
    branch_idx = jnp.clip(branch_idx, 0, len(_RAGGED_K_RUNGS))

    num_tokens, actual_num_q_heads, actual_head_dim = q.shape

    def _branch_0(_):
        out = jnp.zeros((num_tokens, actual_num_q_heads, actual_head_dim), dtype=q.dtype)
        if lse_lane_dense:
            lse = jnp.full((num_tokens, actual_num_q_heads), -jnp.inf, dtype=jnp.float32)
        else:
            lse = jnp.full((num_tokens, actual_num_q_heads, 1), -jnp.inf, dtype=jnp.float32)
        return out, lse

    def _make_rung_branch(rung: int):
        def _branch(_):
            if topk_indices.shape[-1] >= rung:
                sub_topk = topk_indices[:, :rung]
            else:
                sub_topk = jnp.pad(
                    topk_indices,
                    ((0, 0), (0, rung - topk_indices.shape[-1])),
                    constant_values=-1,
                )
            return sparse_ragged_paged_attention(
                q,
                cache_kv_nope,
                cache_kv_rope,
                sub_topk,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=sm_scale,
                k_scale=k_scale,
                gather_and_attention_chunk_size=gather_and_attention_chunk_size,
                attention_kernel_batch_size=attention_kernel_batch_size,
                vmem_limit_bytes=vmem_limit_bytes,
                return_lse=True,
                cache_layout=cache_layout,
                sparsecore_atoms_per_batch=sparsecore_atoms_per_batch,
                sparsecore_gather_wait_mode=sparsecore_gather_wait_mode,
                sparsecore_gather_pipelined=sparsecore_gather_pipelined,
                is_dcp_mode=True,
                q_pe=q_pe,
                lse_lane_dense=lse_lane_dense,
            )
        return _branch

    branches = [_branch_0] + [_make_rung_branch(r) for r in _RAGGED_K_RUNGS]
    return jax.lax.switch(branch_idx, branches, None)

