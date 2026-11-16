"""Validated FP8 QK/PV math and offset/reuse DMA schedule, ported to main."""
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import get_dtype_packing

def plain_kernel(
    # Prefetch
    kv_lens_ref,  # [max_num_seqs]
    start_end_seq_idx_ref,  # [3] (start_seq_idx, end_seq_idx, num_valid_tokens)
    sem_ids_ref,  # [2] (bi_sem_idx, bo_sem_idx)
    bo_ids_ref,  # [2] seq_idx_start of the in-flight bo DMA per bo sem
    # Input
    attention_sinks_ref,  # float32[num_q_heads]
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # [total_num_pages, page_size * 4, 128]
    cache_kv_rope_hbm_ref,  # [total_num_pages, page_size, rope_dim]
    swa_accumution_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # [max_num_tokens, num_l_heads]
    swa_m_hbm_ref,  # [max_num_tokens, num_l_heads]
    q8_hbm_ref, k8_hbm_ref, qs_hbm_ref, ks_hbm_ref, vs_ref, destination_ref,
    # Output
    o_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    # Scratch
    bkv_nope_x2_ref,  # [2, batch_size, page_size * 4, 128]
    bkv_rope_x2_ref,  # [2, batch_size, page_size, rope_dim]
    bq_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bo_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    bl_x2_ref,  # [2, batch_size, num_l_heads]
    bm_x2_ref,  # [2, batch_size, num_l_heads]
    swa_acc_x2_ref,  # [2, batch_size, num_q_heads, head_dim]
    sems,  # [7, 2]
    q8_buf, k8_buf, qs_buf, ks_buf, fp8_sems,
    *,
    sm_scale: float,
    batch_size: int = 1,
    schedule: str = "interleave",
    ablation: str = "full",
    pv_mode: str = "prequant",
    exp_shift: bool = False,
    input_offset: int = 0,
    output_offset: int = 0,
    chunk_tokens: int = 512,
):
    assert ablation == "full" and pv_mode == "prequant" and not exp_shift
    assert q_hbm_ref.shape[1:] == o_hbm_ref.shape[1:]

    _, num_q_heads, head_dim = q_hbm_ref.shape
    num_tokens = chunk_tokens
    assert kv_lens_ref.shape[0] == num_tokens
    predequantized = cache_kv_rope_hbm_ref is None
    assert predequantized and cache_kv_nope_hbm_ref.shape[1] <= 512
    # Keep exact 512-column QK/PV and softmax reduction shape.
    bkv_sz = 512

    assert q_hbm_ref.dtype == jnp.float8_e4m3fn
    q_dtype = o_hbm_ref.dtype
    q_packing = get_dtype_packing(q_dtype)
    # Validate against the KV dtype.
    assert o_hbm_ref.dtype == q_dtype

    assert head_dim % 128 == 0
    assert num_q_heads % q_packing == 0

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    num_valid_tokens = start_end_seq_idx_ref[2]

    batch_start_seq_idx = start_seq_idx + pl.program_id(0) * batch_size
    batch_end_seq_idx = batch_start_seq_idx + batch_size - 1

    def flash_attention_step1_qk(
            q,  # [bq_sz * num_q_heads, head_dim]
            kv,  # [bkv_sz, head_dim] <- Correspond to data from bkv_*_x2_ref
    ):
        if ablation in ("qk_identity", "mxu_identity"):
            return q.astype(jnp.float32)
        if ablation in ("no_qk", "no_mxu"):
            return q[:, :1].astype(jnp.float32) + kv[:1, :].astype(jnp.float32)
        if ablation == "cast_bf16":
            q, kv = q.astype(jnp.bfloat16), kv.astype(jnp.bfloat16)
        assert len(q.shape) == 2
        assert len(kv.shape) == 2
        assert q.shape[0] % num_q_heads == 0
        assert q.shape[1] == head_dim
        assert kv.shape == (bkv_sz, head_dim)

        return jnp.einsum("nd,md->nm",
                          q,
                          kv,
                          preferred_element_type=jnp.float32)

    def flash_attention_step1_softmax(
            s,
            kv_len,  # scalar
            swa_m,  # [bq_sz * num_q_heads],
            swa_l,  # [bq_sz * num_q_heads],
            attention_sinks,  # [num_q_heads]
    ):
        if ablation == "no_softmax":
            return s, jnp.ones_like(swa_m), jnp.ones_like(swa_l)
        # Follow FlashAttention-2 forward pass.
        s *= sm_scale
        k_span = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        s = jnp.where(k_span < kv_len, s, jnp.finfo(s.dtype).min)

        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = swa_m
        m_curr = jnp.maximum(m_prev, s_rowmax)
        p = jnp.exp((s - m_curr) + jnp.float32(6.104793232414985)) if exp_shift else jnp.exp(s - m_curr)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        if exp_shift:
            p_rowsum *= jnp.float32(1.0 / 448.0)
        l_prev = swa_l
        l_curr = exp_m_diff * l_prev + p_rowsum
        exp_attention_sinks = jnp.exp(attention_sinks - m_curr)
        l = l_curr + exp_attention_sinks  # noqa: E741

        return p, exp_m_diff, l

    def flash_attention_step2_pv(p, value):
        # One V scale for the entire sequence/KV head; no column scaling.
        p8 = (p if exp_shift else p * jnp.float32(448.0)).astype(jnp.float8_e4m3fn)
        v8 = value
        if pv_mode == "cast":
            p8, v8 = p8.astype(jnp.bfloat16), v8.astype(jnp.bfloat16)
        return jnp.einsum("nm,md->nd", p8, v8, preferred_element_type=jnp.float32)

    def flash_attention_step2_finalize(
        pv,
        exp_m_diff,
        swa_acc,
        l_total,
    ):
        o_prev = swa_acc
        acc = exp_m_diff * o_prev + pv * (jnp.float32(1.0 / 448.0))
        out = (lax.div(acc, l_total) if q_dtype == jnp.float32 else
               (acc * pl.reciprocal(l_total, approx=True)).astype(q_dtype))
        return out

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, *, wait=False):
        if k8_hbm_ref is not None:
            _fp8_copy(k8_hbm_ref, k8_buf, 1, seq_idx_start - start_seq_idx, bkv_sem_idx, wait)
        sem_nope = sems.at[0, bkv_sem_idx]
        sem_rope = sems.at[6, bkv_sem_idx]

        bkv_nope_vmem_ref = bkv_nope_x2_ref.at[bkv_sem_idx, :]
        bkv_rope_vmem_ref = (None if predequantized else
                             bkv_rope_x2_ref.at[bkv_sem_idx, :])

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
            if not predequantized:
                _async_copy(
                    cache_kv_rope_hbm_ref.at[pl.ds(page_idx_start,
                                                   batch_size)],
                    bkv_rope_vmem_ref,
                    sem_rope,
                    wait,
                )
        else:
            dst_nope = bkv_nope_vmem_ref
            _async_copy(src=dst_nope, dst=dst_nope, sem=sem_nope, wait=True)
            if not predequantized:
                dst_rope = bkv_rope_vmem_ref
                _async_copy(src=dst_rope,
                            dst=dst_rope,
                            sem=sem_rope,
                            wait=True)

    def _fp8_copy(src, buf, sem_index, seq_start, bi, wait):
        dst = buf.at[bi, :]
        sem = fp8_sems.at[sem_index, bi]
        if wait:
            _async_copy(dst, dst, sem, True)
        else:
            _async_copy(src.at[pl.ds(seq_start, batch_size)], dst, sem, False)

    def _fetch_bq_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        _fp8_copy(q8_hbm_ref, q8_buf, 0, seq_idx_start + input_offset, bq_sem_idx, wait)

    def _send_bo_batch(seq_idx_start, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        sz = jnp.clip(num_valid_tokens - seq_idx_start, 1, batch_size)
        vmem_ref = bo_x2_ref.at[bo_sem_idx, pl.ds(0, sz)]

        if not wait:
            # Remember where this DMA started so its wait can rebuild the same
            # `sz`; a mismatch here deadlocks on the DMA semaphore.
            bo_ids_ref[bo_sem_idx] = seq_idx_start
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[pl.ds(seq_idx_start + output_offset, sz)],
                sem,
                wait,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _fetch_swa_batch(seq_idx_start, bq_sem_idx, *, wait=False):
        sem_acc = sems.at[3, bq_sem_idx]
        sem_l = sems.at[4, bq_sem_idx]
        sem_m = sems.at[5, bq_sem_idx]

        if not wait:
            _async_copy(
                swa_accumution_hbm_ref.at[pl.ds(seq_idx_start + input_offset, batch_size)],
                swa_acc_x2_ref.at[bq_sem_idx, :],
                sem_acc,
                wait=False,
            )
            _async_copy(
                swa_l_hbm_ref.at[pl.ds(seq_idx_start + input_offset, batch_size)],
                bl_x2_ref.at[bq_sem_idx, :],
                sem_l,
                wait=False,
            )
            _async_copy(
                swa_m_hbm_ref.at[pl.ds(seq_idx_start + input_offset, batch_size)],
                bm_x2_ref.at[bq_sem_idx, :],
                sem_m,
                wait=False,
            )

        else:
            dst_acc = swa_acc_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_acc, dst=dst_acc, sem=sem_acc, wait=True)

            dst_l = bl_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_l, dst=dst_l, sem=sem_l, wait=True)

            dst_m = bm_x2_ref.at[bq_sem_idx, :]
            _async_copy(src=dst_m, dst=dst_m, sem=sem_m, wait=True)

    def start_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx)

    def wait_fetch_bkv_batch(seq_idx_start, bkv_sem_idx):
        return _fetch_bkv_batch(seq_idx_start, bkv_sem_idx, wait=True)

    def start_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_bq_batch(seq_idx_start, bq_sem_idx):
        return _fetch_bq_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx)

    def wait_fetch_swa_batch(seq_idx_start, bq_sem_idx):
        return _fetch_swa_batch(seq_idx_start, bq_sem_idx, wait=True)

    def start_send_bo_batch(seq_idx_start, bo_sem_idx):
        return _send_bo_batch(seq_idx_start, bo_sem_idx)

    def wait_send_bo_batch(bo_sem_idx):
        # `sz` must match the started DMA exactly, so recover its
        # seq_idx_start rather than using the current step's.
        old_seq_idx_start = bo_ids_ref[bo_sem_idx]

        @pl.when(old_seq_idx_start >= 0)
        def _():
            _send_bo_batch(old_seq_idx_start, bo_sem_idx, wait=True)

    def load_bq(bq_sem_idx, batch_idx):
        q = q8_buf.at[bq_sem_idx, batch_idx][...]
        return q

    def load_bkv(bkv_sem_idx, batch_idx, dsv4_fp8_scale_expand_matrix):
        raw = bkv_nope_x2_ref.at[bkv_sem_idx, batch_idx][...]
        words = raw[:, :128].astype(jnp.uint32)
        bits = jnp.concatenate([(words >> bit).astype(jnp.uint8) for bit in (0,8,16,24)], axis=-1)
        v = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn)
        if v.shape[0] < 512:
            v = jnp.pad(v, ((0, 512 - v.shape[0]), (0, 0)))
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        k_span = lax.broadcasted_iota(jnp.int32, v.shape, 0)
        v = jnp.where(k_span < kv_len, v, 0)
        if pv_mode == "vmem":
            return v
        return v

    def load_swa_output(bq_sem_idx, batch_idx):
        swa_acc = swa_acc_x2_ref[bq_sem_idx, batch_idx, ...]
        swa_l = bl_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        swa_m = bm_x2_ref[bq_sem_idx, batch_idx, :num_q_heads][..., None]
        return swa_acc, swa_l, swa_m

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
            start_fetch_swa_batch(next_seq_idx, next_bi_sem_idx)
            start_fetch_bkv_batch(next_seq_idx, next_bi_sem_idx)

        bo_sem_idx = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)
        attention_sinks = attention_sinks_ref[...][..., None]
        dsv4_fp8_scale_expand_matrix = None

        prev_p = None
        prev_bkv = None
        prev_exp_m_diff = None
        prev_l = None
        prev_swa_acc = None
        prev_pv = None

        # Wait for cur blocks if not ready yet
        wait_fetch_bq_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_swa_batch(batch_start_seq_idx, bi_sem_idx)
        wait_fetch_bkv_batch(batch_start_seq_idx, bi_sem_idx)

        @pl.when(pl.program_id(0) >= 2)
        def _wait_send():
            wait_send_bo_batch(bo_sem_idx)

        for batch_idx in range(batch_size):
            bkv = load_bkv(bi_sem_idx, batch_idx, dsv4_fp8_scale_expand_matrix)
            bq = load_bq(bi_sem_idx, batch_idx)

            if schedule == "interleave" and prev_pv is not None:
                # Artificial dependency to force MXU/VPU interleaving by limiting LLO's QK runahead
                # We use jnp.where to prevent XLA from optimizing the dependency away.
                # prev_pv won't be inf in practice. Depending on the raw PV
                # result releases the next QK before PV's VPU-only finalize.
                bq = jnp.where(prev_pv == jnp.inf,
                               jnp.asarray(jnp.inf, dtype=bq.dtype), bq)

            swa_acc, swa_l, swa_m = load_swa_output(bi_sem_idx, batch_idx)

            s = flash_attention_step1_qk(bq, bkv)

            if prev_p is not None:
                assert prev_bkv is not None
                pv = flash_attention_step2_pv(prev_p, prev_bkv)

            p, exp_m_diff, l = flash_attention_step1_softmax(  # noqa: E741
                s,
                kv_lens_ref[batch_start_seq_idx + batch_idx],
                swa_m,
                swa_l,
                attention_sinks,
            )

            if prev_p is not None:
                assert prev_exp_m_diff is not None
                assert prev_l is not None
                out = flash_attention_step2_finalize(
                    pv,
                    prev_exp_m_diff,
                    prev_swa_acc,
                    prev_l,
                )

                # Store output from acc to bo.
                bo_x2_ref.at[bo_sem_idx, batch_idx - 1][...] = out
                prev_pv = pv

            prev_p = p
            prev_bkv = bkv
            prev_exp_m_diff = exp_m_diff
            prev_l = l
            prev_swa_acc = swa_acc

        # end of pipelining loop
        pv = flash_attention_step2_pv(prev_p, prev_bkv)
        out = flash_attention_step2_finalize(pv, prev_exp_m_diff,
                                             prev_swa_acc, prev_l)
        bo_x2_ref.at[bo_sem_idx, batch_size - 1][...] = out
        start_send_bo_batch(batch_start_seq_idx, bo_sem_idx)

    ### ------- Kernel start ------- ###

    @pl.when(batch_start_seq_idx == start_seq_idx)
    def prologue():
        start_fetch_bq_batch(batch_start_seq_idx, 0)
        start_fetch_swa_batch(batch_start_seq_idx, 0)
        start_fetch_bkv_batch(batch_start_seq_idx, 0)

    process()

    @pl.when(batch_end_seq_idx == end_seq_idx - 1)
    def epilogue():
        wait_send_bo_batch(0)

        @pl.when(pl.num_programs(0) >= 2)
        def _wait_1():
            wait_send_bo_batch(1)
