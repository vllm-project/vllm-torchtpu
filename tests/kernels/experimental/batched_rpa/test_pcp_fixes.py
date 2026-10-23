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
"""CPU unit tests for Batched RPA PCP kernel bug fixes."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    bref_override,
    configs,
    flash_attention,
    kernel,
    schedule,
    schedule_cp,
    utils,
    wrapper,
)


@pytest.fixture
def tpu_7x_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


@pytest.mark.cpu_test
def test_flash_attention_multi_k_block_preserves_softmax_denominator(tpu_7x_info):
    """Multi-K-block Q tile must preserve l_prev when is_last_k resets m_prev.

    In flash_attention_qk, the first loop resets m_prev = -inf when
    is_last_k[step, b_idx] == 1. Keying the l_prev term on alpha_list[b_idx] == 0
    (computed before m_prev is overwritten) preserves the accumulated softmax
    denominator on the final K block.
    """
    kv_heads = 2
    q_per_kv = 4
    bq_c_sz = 4
    tq = q_per_kv * bq_c_sz
    bkv_sz = 128
    head_dim = 128

    model_cfgs = configs.ModelConfigs(
        num_q_heads=kv_heads * q_per_kv,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        mask_value=-1e30,
        sm_scale=0.125,
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=1,
        page_size=128,
        total_q_tokens=bq_c_sz,
        num_page_indices=2,
        dtype_q=jnp.float32,
        dtype_kv=jnp.float32,
        dtype_out=jnp.float32,
    )
    block_cfgs = configs.BlockSizes(
        bq_sz=bq_c_sz,
        bq_c_sz=bq_c_sz,
        bkv_sz=bkv_sz,
        batch_size=1,
        n_buffer=2,
    )
    cfgs = configs.RpaConfigs(
        block=block_cfgs,
        model=model_cfgs,
        serve=serve_cfgs,
        vmem_limit_bytes=64 * 1024 * 1024,
        mode=configs.RpaCase.MIXED,
    )

    rng = np.random.default_rng(42)
    q = jnp.asarray(rng.standard_normal((1, kv_heads, tq, head_dim), dtype=np.float32))
    # Block 0 has slightly higher logits than block 1 so alpha == 1.0 on step 1
    # and dropping l_prev on step 1 causes a >3x error in o / l.
    k0 = jnp.asarray(
        rng.standard_normal((1, kv_heads, bkv_sz, head_dim), dtype=np.float32) + 0.5
    )
    k1 = jnp.asarray(
        rng.standard_normal((1, kv_heads, bkv_sz, head_dim), dtype=np.float32) - 0.5
    )
    v0 = jnp.asarray(
        rng.standard_normal((1, kv_heads, bkv_sz, head_dim), dtype=np.float32)
    )
    v1 = jnp.asarray(
        rng.standard_normal((1, kv_heads, bkv_sz, head_dim), dtype=np.float32)
    )

    mask = [jnp.ones((kv_heads, tq, bkv_sz), dtype=jnp.bool_)]
    is_last_k = jnp.array([[0], [1]], dtype=jnp.int32)

    m_prev = jnp.full((kv_heads, tq, 128), -jnp.inf, dtype=jnp.float32)
    l_prev = jnp.zeros((kv_heads, tq, 128), dtype=jnp.float32)
    o_prev = jnp.zeros((kv_heads, tq, head_dim), dtype=jnp.float32)

    # Step 0: non-final K block (is_last_k == 0).
    p0, alpha0, _, _, m_prev, l_prev = flash_attention.flash_attention_qk_softmax(
        step=jnp.int32(0),
        q=q,
        k=k0,
        m_prev=m_prev,
        l_prev=l_prev,
        is_last_k=is_last_k,
        custom_mask=mask,
        cfgs=cfgs,
        bq_start=0,
    )
    _, o_prev = flash_attention.flash_attention_pv(
        p=p0,
        v=v0,
        alpha_list=alpha0,
        o_prev=o_prev,
        cfgs=cfgs,
        is_last_k=is_last_k,
        step=jnp.int32(0),
    )
    l_prev_before_last = l_prev

    # Step 1: final K block (is_last_k == 1 -> m_prev resets to -inf in loop 1).
    p1, alpha1, m_next1, l_next1, m_after, l_after = (
        flash_attention.flash_attention_qk_softmax(
            step=jnp.int32(1),
            q=q,
            k=k1,
            m_prev=m_prev,
            l_prev=l_prev,
            is_last_k=is_last_k,
            custom_mask=mask,
            cfgs=cfgs,
            bq_start=0,
        )
    )
    o_next1, o_after = flash_attention.flash_attention_pv(
        p=p1,
        v=v1,
        alpha_list=alpha1,
        o_prev=o_prev,
        cfgs=cfgs,
        is_last_k=is_last_k,
        step=jnp.int32(1),
    )

    assert jnp.all(jnp.isneginf(m_after))
    assert jnp.all(l_after == 0.0)
    assert jnp.all(o_after == 0.0)

    # Verify the old `m_prev == -inf` check (evaluated after m_prev is reset to
    # m_after == -inf) would zero out alpha1 * l_prev_before_last and fail.
    buggy_term_prev = jnp.where(
        m_after == -jnp.inf, 0.0, alpha1[0] * l_prev_before_last
    )
    assert jnp.all(buggy_term_prev == 0.0)
    assert jnp.all(alpha1[0] * l_prev_before_last > 0.0)

    actual_out = o_next1[0] / utils.broadcast_minor(l_next1[0], o_next1[0].shape)

    # Reference single-pass softmax across [k0, k1].
    k_cat = jnp.concatenate([k0[0], k1[0]], axis=1)
    v_cat = jnp.concatenate([v0[0], v1[0]], axis=1)
    logits = jnp.einsum("hqd,hkd->hqk", q[0], k_cat) * model_cfgs.sm_scale
    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    probs_unnorm = jnp.exp(logits - logits_max)
    denom_ref = jnp.sum(probs_unnorm, axis=-1, keepdims=True)
    expected_out = jnp.einsum("hqk,hkd->hqd", probs_unnorm / denom_ref, v_cat)

    np.testing.assert_allclose(
        np.asarray(l_next1[0, :, :, :1]),
        np.asarray(denom_ref),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(actual_out),
        np.asarray(expected_out),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.cpu_test
def test_hnd_paged_new_kv_fetch_val_and_wait_lanes_in_tokens(monkeypatch, tpu_7x_info):
    """Under SEQ_ALONG_LANE + paged_new_kv=True, fetch_val is in tokens (128), not pages (1)."""
    page_size = 128
    model_cfgs = configs.ModelConfigs(
        num_q_heads=8,
        num_kv_heads=2,
        head_dim=256,
        mask_value=-1e30,
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=2,
        page_size=page_size,
        total_q_tokens=256,
        num_page_indices=8,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e4m3fn,
        dtype_out=jnp.bfloat16,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        cp_group_size=2,
        paged_new_kv=True,
        attention_scope=configs.AttentionScope.NEW_TOKENS_ONLY,
        return_lse=True,
    )
    block_cfgs = configs.BlockSizes(
        bq_sz=128,
        bq_c_sz=128,
        bkv_sz=256,
        batch_size=1,
        n_buffer=2,
    )
    cfgs = configs.RpaConfigs(
        block=block_cfgs,
        model=model_cfgs,
        serve=serve_cfgs,
        vmem_limit_bytes=64 * 1024 * 1024,
        mode=configs.RpaCase.MIXED,
    )

    sched_shape = schedule_cp.CPMetadataComputer.get_rpa_schedule(cfgs)
    sched_init = jax.tree.map(lambda sd: jnp.zeros(sd.shape, sd.dtype), sched_shape)
    dma_sem_init = jnp.zeros((1,), dtype=jnp.int32)

    class _NoopCopy:
        def start(self):
            pass

        def wait(self):
            pass

    monkeypatch.setattr(pltpu, "make_async_copy", lambda *a, **kw: _NoopCopy())

    cu_q_lens = jnp.array([0, 128, 256], dtype=jnp.int32)
    q_positions = jnp.array([0, 384], dtype=jnp.int32)
    global_kv_cache_lens = jnp.array([0, 0], dtype=jnp.int32)
    kv_new_lens = jnp.array([512, 512], dtype=jnp.int32)
    update_kv_cache = jnp.array([0, 1], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 2], dtype=jnp.int32)
    cp_rank = jnp.array([0], dtype=jnp.int32)

    def run_schedule(refs):
        (
            sched_ref,
            sem_ref,
            cu_ref,
            qpos_ref,
            cache_ref,
            new_ref,
            upd_ref,
            dist_ref,
            rank_ref,
        ) = refs
        comp = schedule_cp.CPMetadataComputer(
            schedule=sched_ref,
            schedule_hbm_ref=sched_ref,
            dma_sem=sem_ref,
            cfgs=cfgs,
            extra_refs=(rank_ref,),
        )
        carry = comp.compute(
            cu_q_lens_ref=cu_ref,
            q_positions_ref=qpos_ref,
            global_kv_cache_lens_ref=cache_ref,
            kv_new_lens_ref=new_ref,
            update_kv_cache_ref=upd_ref,
            distribution_ref=dist_ref,
        )
        steps = pl.cdiv(carry.count, cfgs.batch_size)
        schedule._compute_waits(sched_ref, 0, steps, cfgs=cfgs)

    sched_out, *_ = pl.run_state(run_schedule)(
        (
            sched_init,
            dma_sem_init,
            cu_q_lens,
            q_positions,
            global_kv_cache_lens,
            kv_new_lens,
            update_kv_cache,
            distribution,
            cp_rank,
        )
    )

    # bkv_sz=256 spans 2 pages of 128 tokens -> active slots 0..1 must record
    # 128 tokens (not 1 page); spillover slot 2 has 0 tokens when page-aligned.
    dma_kv_new_4d = sched_out.dma_kv_new.data.reshape(*sched_out.dma_kv_new.shape, 5)
    active_pages = cfgs.bkv_sz // page_size
    for page_slot in range(active_pages):
        flags = int(dma_kv_new_4d[0, 0, page_slot, 4])
        fetch_val = flags & 0xFFFF
        assert fetch_val == page_size, f"slot {page_slot}: {fetch_val} != {page_size}"
    assert (int(dma_kv_new_4d[0, 0, active_pages, 4]) & 0xFFFF) == 0

    expected_wait_lanes = (cfgs.bkv_sz * cfgs.kv_bytes_per_token) // (128 * 4)
    assert int(sched_out.total_wait_kv_in[0]) == expected_wait_lanes


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "num_tokens,dtype,expected_alloc",
    [
        (1, jnp.bfloat16, 16),
        (5, jnp.bfloat16, 16),
        (9, jnp.bfloat16, 16),
        (15, jnp.bfloat16, 16),
        (16, jnp.bfloat16, 16),
        (17, jnp.bfloat16, 32),
        (1, jnp.float32, 8),
        (9, jnp.float32, 16),
    ],
)
def test_packed_lse_non_multiple_of_16_alignment_and_wait_bytes(
    monkeypatch, tpu_7x_info, num_tokens, dtype, expected_alloc
):
    """Packed LSE buffer aligns alloc_tokens to num_sublanes * packing (16 for bf16, 8 for fp32)."""
    num_q_heads = 2
    num_kv_heads = 2
    head_dim = 128
    page_size = 128

    queries = jnp.zeros((num_tokens, num_q_heads, head_dim), dtype=dtype)
    keys = jnp.zeros((num_tokens, num_kv_heads, head_dim), dtype=dtype)
    values = jnp.zeros((num_tokens, num_kv_heads, head_dim), dtype=dtype)
    kv_cache_shape = wrapper.get_kv_cache_shape(
        total_num_pages=4,
        page_size=page_size,
        actual_num_kv_heads=num_kv_heads,
        actual_head_dim=head_dim,
        kv_dtype=dtype,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    kv_cache = jnp.zeros(kv_cache_shape, dtype=dtype)
    kv_lens = jnp.array([page_size], dtype=jnp.int32)
    page_indices = jnp.array([0, 1], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, num_tokens], dtype=jnp.int32)
    distribution = jnp.array([0, 0, 1], dtype=jnp.int32)

    captured_lse_shapes: list[tuple[int, ...]] = []
    captured_cfgs: list[configs.RpaConfigs] = []

    monkeypatch.setattr(schedule, "generate_rpa_metadata", lambda *a, **kw: None)

    def fake_rpa_kernel(*args, cfgs, **kwargs):
        lse_hbm_in = args[11]
        o_hbm_in = args[8]
        kv_cache_in = args[10]
        captured_lse_shapes.append(lse_hbm_in.shape)
        captured_cfgs.append(cfgs)
        return o_hbm_in, kv_cache_in, jnp.zeros_like(lse_hbm_in)

    monkeypatch.setattr(kernel, "rpa_kernel", fake_rpa_kernel)

    out, _, lse = wrapper.ragged_paged_attention(
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        page_indices=page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        return_lse=True,
        cp_group_size=2,
        attention_scope=configs.AttentionScope.CACHE_ONLY,
        pcp_ring_axis_name="pcp",
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )

    assert captured_cfgs[0].serve.is_packed_lse is True
    assert captured_lse_shapes[0] == (num_kv_heads, expected_alloc, 128)
    assert out.shape == (num_tokens, num_q_heads, head_dim)
    assert lse.shape == (num_tokens, num_q_heads)

    # Also check BatchingLSERef.copy_out and wait_out transfer/wait byte parity.
    cfgs_decode = captured_cfgs[0]
    copy_bytes: list[int] = []

    class _FakeCopy:
        def __init__(self, src):
            self.nbytes = int(np.prod(src.shape)) * jnp.dtype(src.dtype).itemsize

        def start(self):
            copy_bytes.append(self.nbytes)

        def wait(self):
            copy_bytes.append(self.nbytes)

    monkeypatch.setattr(pltpu, "make_async_copy", lambda src, dst, sem: _FakeCopy(src))

    fake_sched = SimpleNamespace(
        is_last_k=jnp.array([[1] * cfgs_decode.batch_size], dtype=jnp.int32),
        get_dma_q=lambda block_idx, b: ((0, num_tokens) if b == 0 else (0, 0)),
        total_wait_lse_out=jnp.array(
            [num_tokens * num_kv_heads * cfgs_decode.lse_rows_per_token],
            dtype=jnp.int32,
        ),
    )

    class _SliceableArray:
        def __init__(self, arr):
            self.arr = arr
            self.shape = arr.shape
            self.dtype = arr.dtype

        @property
        def at(self):
            return self

        def __getitem__(self, idx):
            def _conv(s):
                if hasattr(s, "start") and hasattr(s, "size"):
                    return slice(int(s.start), int(s.start) + int(s.size))
                return s

            if isinstance(idx, tuple):
                idx = tuple(_conv(i) for i in idx)
            else:
                idx = _conv(idx)
            return _SliceableArray(self.arr[idx])

        def reshape(self, shape):
            return _SliceableArray(self.arr.reshape(shape))

    vmem_arr = _SliceableArray(
        jnp.zeros(
            (1, cfgs_decode.batch_size, num_kv_heads, expected_alloc, 128), dtype=dtype
        )
    )
    lse_hbm_arr = _SliceableArray(
        jnp.zeros((num_kv_heads, expected_alloc, 128), dtype=dtype)
    )
    sem_arr = _SliceableArray(jnp.zeros((1,), dtype=jnp.int32))
    dummy_lse_ref = SimpleNamespace(
        cfgs=cfgs_decode,
        current_copy_out_slot=0,
        current_wait_out_slot=0,
        sem_sends=sem_arr,
        window_ref=vmem_arr,
    )
    bref_override.BatchingLSERef.copy_out(
        dummy_lse_ref, (lse_hbm_arr, fake_sched), (0,)
    )
    sent_bytes = sum(copy_bytes)
    copy_bytes.clear()
    bref_override.BatchingLSERef.wait_out(
        dummy_lse_ref, (lse_hbm_arr, fake_sched), (0,)
    )
    waited_bytes = sum(copy_bytes)
    assert waited_bytes == sent_bytes
    assert waited_bytes == (num_tokens * num_kv_heads * 128 * jnp.dtype(dtype).itemsize)
