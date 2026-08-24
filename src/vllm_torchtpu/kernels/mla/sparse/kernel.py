"""TPU-Friendly MLA Ragged Paged Attention kernel - GLM5.2 DeepSeek Sparse Attention."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla.sparse import dsa_gather

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


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


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
    kv_packing: int | None = None,
):
    if kv_packing is None:
        kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


_GATHER_PAGE_CHUNK = 128


def _gather_page_ids_kernel(windows_ref, logical_ref, out_ref, *, num_chunks):
    logical = logical_ref[...]  # i32[block_tokens, topk]
    out = jnp.zeros_like(logical)
    for c in range(num_chunks):
        window_chunk = windows_ref[:, c * _GATHER_PAGE_CHUNK:(
            c + 1) * _GATHER_PAGE_CHUNK]  # i32[block_tokens, 128]
        local = logical - c * _GATHER_PAGE_CHUNK
        gathered = jnp.take_along_axis(window_chunk,
                                       jnp.clip(local, 0,
                                                _GATHER_PAGE_CHUNK - 1),
                                       axis=1)
        out = jnp.where((local >= 0) & (local < _GATHER_PAGE_CHUNK), gathered,
                        out)
    out_ref[...] = out


def gather_page_ids(
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    seq_page_ids: jax.
    Array,  # i32[num_tokens, topk]  (logical page within seq)
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
        page_table = jnp.pad(page_table,
                             ((0, 0), (0, padded_pps - pages_per_seq)))
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
            grid=(padded_tokens // block_tokens, ),
        ),
        out_shape=jax.ShapeDtypeStruct((padded_tokens, topk), jnp.int32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", ),
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
    # [num_tokens, topk * TILE_SUBROWS, TILE_LANE_BYTES] (gathered, raw tiles)
    cache_kv_nope_hbm_ref,
    cache_kv_rope_hbm_ref,  # [num_tokens, topk, rope_dim] (gathered)
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, topk * TILE_SUBROWS, TILE_LANE_BYTES]
    bkv_rope_x2_ref,  # [2, batch_size, topk, rope_dim]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [4, 2]
    *,
    sm_scale: float,
    k_scale: float = 1.0,
    batch_size: int = 1,
):
    assert q_hbm_ref.shape == o_hbm_ref.shape

    num_tokens, num_q_heads, head_dim = q_hbm_ref.shape
    nope_dim = dsa_gather.TILE_SUBROWS * dsa_gather.TILE_LANE_BYTES
    assert kv_lens_ref.shape[0] == num_tokens
    bkv_sz = cache_kv_nope_hbm_ref.shape[1] // dsa_gather.TILE_SUBROWS

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

    def attention_step1_qk_softmax(
            q,  # [num_q_heads, head_dim]
            kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
            kv_len,  # i32[] number of valid (non "-1") top-k rows in kv
    ):
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)

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

        return p, l_sum

    def attention_step2_pv(
        p,
        kv,
        l_sum,
    ):
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)

        # We use `k_scale` for the PV step as well since MLA uses a shared
        # compressed latent for k/v and k_scale ~= v_scale.
        if k_scale != 1.0:
            pv *= k_scale

        out = (lax.div(pv, l_sum) if q_dtype == jnp.float32 else
               (pv * pl.reciprocal(l_sum, approx=True)).astype(q_dtype))
        return out

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

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
        else:
            _async_copy(src=bq_vmem_ref, dst=bq_vmem_ref, sem=sem, wait=True)

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
        return q

    def load_bkv(bkv_sem_idx, batch_idx):
        # The gather nope is (4, 128) fp8, reshape it to (1, 512) fp8.
        bkv_nope = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        bkv_nope = bkv_nope.reshape(bkv_sz, nope_dim)
        bkv_nope = pltpu.bitcast(bkv_nope, jnp.float8_e4m3fn)

        bkv_rope = bkv_rope_x2_ref.at[bkv_sem_idx, batch_idx][...]
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

    def process():

        def get_next_seq_ids(seq_idx, bi_sem_idx):
            next_seq_idx = seq_idx + batch_size
            next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bi_sem_idx

        bi_sem_idx = sem_ids_ref[0]
        next_seq_idx, next_bi_sem_idx = get_next_seq_ids(
            batch_start_seq_idx, bi_sem_idx)

        # Prefetch next seq
        @pl.when(next_seq_idx < end_seq_idx)
        def prefetch_next_seq():
            sem_ids_ref[0] = next_bi_sem_idx
            start_fetch_bq_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

        prev_p = None
        prev_bkv = None
        prev_l_sum = None
        prev_out = None

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

        for batch_idx in range(batch_size):
            kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
            bkv = load_bkv(bi_sem_idx, batch_idx)
            bq = load_bq(bi_sem_idx, batch_idx)

            if prev_out is not None:
                # Artificial dependency to force MXU/VPU interleaving by limiting LLO's QK runahead
                # We use jnp.where to prevent XLA from optimizing the dependency away.
                # prev_out won't be inf in practice.
                bq = jnp.where(prev_out == jnp.inf, prev_out, bq)

            p, l_sum = attention_step1_qk_softmax(bq, bkv, kv_len)

            if prev_p is not None:
                assert prev_bkv is not None
                assert prev_l_sum is not None
                out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
                prev_out = out

            prev_p = p
            prev_bkv = bkv
            prev_l_sum = l_sum

        # end of pipelining loop
        out = attention_step2_pv(prev_p, prev_bkv, prev_l_sum)
        bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        start_fetch_bkv_batch(batch_start_seq_idx, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        # The first argument "0" for seq_idx_start does not matter here.
        wait_send_bo_batch(0, 0)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            # The first argument "0" for seq_idx_start does not matter here.
            wait_send_bo_batch(0, 1)

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
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # [total_num_pages, page_size, 4, 128]
    cache_kv_rope: jax.Array,  # [total_num_pages, page_size // 4, 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    *,
    sm_scale: float = 1.0,
    k_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int | None = None,
    attention_kernel_batch_size: int = 16,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
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

  Returns:
    The output of attention.
  """
    # Opaque uint8 tiles (see dsa_gather.py): nope encodes 512 fp8 values
    # (gathered as raw (4, 128) tiles, then folded to 512 lanes and dequantized
    # in-kernel with the per-tensor k_scale), rope encodes 64 bf16 as high/low
    # byte planes (dequantized by the shim).
    assert cache_kv_nope.dtype == jnp.uint8
    assert cache_kv_rope.dtype == jnp.uint8
    if gather_and_attention_chunk_size is None:
        gather_and_attention_chunk_size = q.shape[0]

    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    head_dim = q.shape[-1]

    _, page_size, _, _ = cache_kv_nope.shape

    _, num_q_heads, _ = q.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0

    def run_mla_kernel(
        q: jax.Array,  # [max_num_tokens, num_q_heads, head_dim]
        # [num_tokens, topk * TILE_SUBROWS, TILE_LANE_BYTES] (gathered raw tiles)
        cache_kv_nope: jax.Array,
        cache_kv_rope: jax.Array,  # [num_tokens, topk, rope_dim] (gathered)
        kv_lens: jax.Array,  # i32[max_num_tokens]
        start_seq_idx: jax.Array,  # i32
        end_seq_idx: jax.Array,  # i32
        kernel_batch_size: int,
    ):
        batch_size = kernel_batch_size
        end_seq_idx = jnp.maximum(start_seq_idx, end_seq_idx)
        grid = (cdiv(end_seq_idx - start_seq_idx, batch_size), )
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),  # q
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_nope
            pl.BlockSpec(memory_space=pltpu.HBM),  # cache_kv_rope
        ]

        out_specs = pl.BlockSpec(memory_space=pltpu.HBM)  # o

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
            (2, batch_size, num_q_heads, head_dim),
            q.dtype,
        )

        bo_double_buf = bq_double_buf

        scratch_shapes = [
            bkv_nope_double_buf,
            bkv_rope_double_buf,
            bq_double_buf,
            bo_double_buf,  # Double buffering for output block.
            # Semaphores for double buffering of bkv_nope, bq, bo, bkv_rope.
            pltpu.SemaphoreType.DMA((4, 2)),
        ]

        scalar_prefetches = (
            kv_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            # (bi_sem_idx, bo_sem_idx)
            jnp.zeros((2, ), jnp.int32),
        )

        scope_name = f"MLA-p_{cache_kv_rope.shape[1]}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel,
                    sm_scale=sm_scale,
                    k_scale=k_scale,
                    batch_size=batch_size,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary", ),
                    vmem_limit_bytes=vmem_limit_bytes,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={
                    # 3 = q; operand indices count the scalar prefetches
                    # (DSV4's "4" also counted attention_sinks).
                    3: 0,  # Alias output activation with q
                },
                name=scope_name,
            ))
        return kernel(
            *scalar_prefetches,
            q,
            cache_kv_nope,
            cache_kv_rope,
        )

    tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
    seq_ids_segment = jnp.repeat(jnp.arange(max_num_seqs),
                                 tokens_per_seq,
                                 total_repeat_length=q.shape[0])
    assert topk_indices is not None
    # TODO: skip gather for padding tokens in topk_indices.
    kv_lens = jnp.sum(topk_indices != -1, axis=-1)

    seq_page_ids = topk_indices // page_size
    token_offset = topk_indices % page_size
    topk = topk_indices.shape[-1]
    page_ids = gather_page_ids(page_indices, seq_page_ids, seq_ids_segment,
                               max_num_seqs)

    # For the "-1" padding elements in topk_indices, we scatter the corresponding
    # page_ids and token_offset to avoid gather memory access hotspotting.
    is_padding = topk_indices == -1
    total_num_pages = cache_kv_nope.shape[0]
    flat_element_index = jnp.arange(q.shape[0] * topk,
                                    dtype=jnp.int32).reshape(q.shape[0], topk)
    # 104729 and 15485863 are randomly chosen large prime numbers.
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(
        is_padding,
        scattered_token_offset,
        token_offset,
    )

    assert page_ids.shape == (q.shape[0], topk)

    # TODO: handle the case where q.shape[0] is not divisible by
    # gather_and_attention_chunk_size.
    assert q.shape[0] % gather_and_attention_chunk_size == 0
    num_chunks = q.shape[0] // gather_and_attention_chunk_size

    for i in range(num_chunks):
        start_pos = i * gather_and_attention_chunk_size
        end_pos = start_pos + gather_and_attention_chunk_size
        indices = (page_ids[start_pos:end_pos, ...] * page_size +
                   token_offset[start_pos:end_pos, ...]).reshape(-1)

        # For prefilling of short sequences (or early in the sequence), there are
        # very few number of KVs in the sequence, so different qs' selected topk
        # would have large overlap. This causes gather read hotspotting. We've seen
        # 30%+ performance degradation compared to the no-duplicate-indices case.
        #
        # TODO: we could consider let the caller (tpu-runner) to sort the sequences
        # based on their lengths. For the sequences-segment below certain length,
        # we use a different kernel (dense attention and mask), for the rest of
        # sequences, we use this gather-and-attention kernel.
        gathered_nope_buffer, gathered_rope_buffer = dsa_gather.dsa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
        )
        gathered_nope_buffer = gathered_nope_buffer.reshape(
            gather_and_attention_chunk_size,
            topk * dsa_gather.TILE_SUBROWS,
            dsa_gather.TILE_LANE_BYTES,
        )
        gathered_rope_buffer = gathered_rope_buffer.reshape(
            gather_and_attention_chunk_size, topk, -1)
        # We treat each query token as a one independent sequence, attend to their
        # respective gathered kv tokens in the `gathered_kv_buffer`.
        # -1 in topk_indices is padded elements at the end of each row.
        # Batching
        assert gather_and_attention_chunk_size % attention_kernel_batch_size == 0
        batch_end = (cdiv(
            jnp.minimum(
                cu_q_lens[distribution[2]],
                start_pos + gather_and_attention_chunk_size,
            ),
            attention_kernel_batch_size,
        ) * attention_kernel_batch_size)
        q = run_mla_kernel(
            q,
            gathered_nope_buffer,
            gathered_rope_buffer,
            kv_lens,
            start_seq_idx=start_pos,
            end_seq_idx=batch_end,
            kernel_batch_size=attention_kernel_batch_size,
        )
    return prepare_outputs(
        q, actual_num_q_heads, actual_head_dim
    )  # [max_num_tokens, actual_num_q_heads, actual_head_dim]
