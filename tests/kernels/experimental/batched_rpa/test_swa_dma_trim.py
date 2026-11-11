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
"""TPU tests for the sliding-window KV DMA trim in BatchedRPA."""

import math
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

pytestmark = pytest.mark.tpu_test

_PAGE = 128
_NUM_KV_HEADS, _NUM_Q_HEADS, _HEAD_DIM = 2, 4, 128
_DECODE_BLOCKS = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8)
_PREFILL_BLOCKS = configs.BlockSizes(bq_sz=512, bq_c_sz=16, bkv_sz=256, batch_size=2)


def _sw_start(kv_len: int, q_len: int, sliding_window: int) -> int:
    """Window start of the first query of a decode/mixed block (may be negative)."""
    return kv_len - q_len - sliding_window + 1


def _numpy_windowed_attention(
    q: np.ndarray,
    ks: Sequence[np.ndarray],
    vs: Sequence[np.ndarray],
    kv_lens: Sequence[int],
    q_len: int,
    sliding_window: int | None,
    sm_scale: float,
) -> np.ndarray:
    """Causal (optionally sliding-window) NumPy FP32 softmax reference."""
    group = _NUM_Q_HEADS // _NUM_KV_HEADS
    ref = np.empty_like(q, dtype=np.float32)
    for s, kv_len in enumerate(kv_lens):
        k_rep = np.repeat(ks[s], group, axis=1)
        v_rep = np.repeat(vs[s], group, axis=1)
        for i in range(q_len):
            pos = kv_len - q_len + i
            lo = 0 if sliding_window is None else max(0, pos - sliding_window + 1)
            scores = (
                np.einsum("hd,thd->ht", q[s * q_len + i], k_rep[lo : pos + 1])
                * sm_scale
            )
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            ref[s * q_len + i] = np.einsum("ht,thd->hd", probs, v_rep[lo : pos + 1])
    return ref


def _seed_and_run_step(
    kv_dtype: jnp.dtype,
    ks: Sequence[np.ndarray],
    vs: Sequence[np.ndarray],
    q: np.ndarray,
    kv_lens: Sequence[int],
    q_len: int,
    sliding_window: int | None,
    *,
    page_size: int = _PAGE,
    page_table: np.ndarray | None = None,
    total_pages: int | None = None,
    distribution: tuple[int, int, int] | None = None,
    decode_block_sizes: configs.BlockSizes | None = _DECODE_BLOCKS,
    prefill_block_sizes: configs.BlockSizes | None = _PREFILL_BLOCKS,
    mask_invalid_v: bool | None = None,
    poison_pre_window_nan: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefill history into paged KV cache, optionally NaN-poison pre-window rows,
    run the step for the trailing `q_len` tokens per sequence, and return
    `(kernel_out_f32, numpy_ref_f32)`.
    """
    num_seqs = len(kv_lens)
    pages_per_seq = math.ceil(max(kv_lens) / page_size) + 1
    if page_table is None:
        rng = np.random.default_rng(seed)
        if total_pages is None:
            total_pages = num_seqs * pages_per_seq + 1
        perm = rng.permutation(total_pages)[: num_seqs * pages_per_seq]
        page_table = perm.reshape(num_seqs, pages_per_seq).astype(np.int32)
    else:
        assert total_pages is not None
        page_table = np.asarray(page_table, dtype=np.int32).reshape(num_seqs, -1)

    cache_shape = wrapper.get_kv_cache_shape(
        total_pages,
        page_size,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        kv_dtype,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
    )
    assert cache_shape[1] == page_size

    sm_scale = _HEAD_DIM**-0.5
    rpa_cfg = configs.RpaConfigs(
        block=_DECODE_BLOCKS,
        model=configs.ModelConfigs(
            num_q_heads=_NUM_Q_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            mask_value=-1e9,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
        ),
        serve=configs.ServingConfigs(
            num_seqs=num_seqs,
            page_size=page_size,
            total_q_tokens=num_seqs * q_len,
            num_page_indices=int(page_table.size),
            dtype_q=jnp.bfloat16,
            dtype_kv=kv_dtype,
            dtype_out=jnp.bfloat16,
            kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        ),
        mode=configs.RpaCase.DECODE,
        vmem_limit_bytes=64 * 1024 * 1024,
    )
    assert rpa_cfg.kv_token_align == 1

    common = dict(
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        mask_invalid_v=mask_invalid_v,
    )
    kv_lens_arr = jnp.asarray(kv_lens, jnp.int32)
    page_flat = jnp.asarray(page_table.ravel(), jnp.int32)

    total_kv = sum(kv_lens)
    _, cache = wrapper.ragged_paged_attention(
        jnp.zeros((total_kv, _NUM_Q_HEADS, _HEAD_DIM), jnp.bfloat16),
        jnp.asarray(np.concatenate(ks), kv_dtype),
        jnp.asarray(np.concatenate(vs), kv_dtype),
        jnp.zeros(cache_shape, kv_dtype),
        kv_lens_arr,
        page_flat,
        jnp.asarray(np.cumsum((0, *kv_lens)), jnp.int32),
        jnp.array([0, 0, num_seqs], jnp.int32),
        **common,
    )

    if poison_pre_window_nan and sliding_window is not None:
        pages, offs = [], []
        for s, kv_len in enumerate(kv_lens):
            for pos in range(max(0, _sw_start(kv_len, q_len, sliding_window))):
                pages.append(page_table[s, pos // page_size])
                offs.append(pos % page_size)
        if pages:
            cache = cache.at[np.asarray(pages), np.asarray(offs)].set(jnp.nan)

    new_k = np.concatenate([k[kv_len - q_len :] for k, kv_len in zip(ks, kv_lens)])
    new_v = np.concatenate([v[kv_len - q_len :] for v, kv_len in zip(vs, kv_lens)])
    step_dist = (num_seqs, num_seqs, num_seqs) if distribution is None else distribution
    decode_q_sz = q_len if step_dist[0] > 0 else 1
    out, _ = wrapper.ragged_paged_attention(
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(new_k, kv_dtype),
        jnp.asarray(new_v, kv_dtype),
        cache,
        kv_lens_arr,
        page_flat,
        jnp.asarray(np.arange(num_seqs + 1) * q_len, jnp.int32),
        jnp.asarray(step_dist, jnp.int32),
        decode_query_size=decode_q_sz,
        **common,
    )

    ks_rt = [np.asarray(jnp.asarray(k, kv_dtype).astype(jnp.float32)) for k in ks]
    vs_rt = [np.asarray(jnp.asarray(v, kv_dtype).astype(jnp.float32)) for v in vs]
    q_rt = np.asarray(jnp.asarray(q, jnp.bfloat16).astype(jnp.float32))
    ref = _numpy_windowed_attention(
        q_rt, ks_rt, vs_rt, kv_lens, q_len, sliding_window, sm_scale
    )
    return np.asarray(out, np.float32), ref


def _run_poisoned_decode(
    kv_dtype: jnp.dtype,
    kv_lens: Sequence[int],
    q_len: int,
    sliding_window: int,
    *,
    loud_seqs: Sequence[int] = (),
    decode_block_sizes: configs.BlockSizes | None = _DECODE_BLOCKS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefill clean history, NaN-poison pre-window cache rows, then decode.

    Without the SWA DMA trim, the boundary KV block DMAs pre-window NaNs into
    VMEM; their attention scores are masked, but PV still computes `0 * NaN = NaN`
    (`mask_invalid_v` defaults to `False`). With the trim they are never copied.
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(kv_lens)
    ks, vs = [], []
    for s, kv_len in enumerate(kv_lens):
        scale = 8.0 if s in loud_seqs else 0.5
        k = (rng.standard_normal((kv_len, _NUM_KV_HEADS, _HEAD_DIM)) * scale).astype(
            np.float32
        )
        v = (rng.standard_normal((kv_len, _NUM_KV_HEADS, _HEAD_DIM)) * scale).astype(
            np.float32
        )
        ks.append(k)
        vs.append(v)
    q = (rng.standard_normal((num_seqs * q_len, _NUM_Q_HEADS, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )
    return _seed_and_run_step(
        kv_dtype,
        ks,
        vs,
        q,
        kv_lens,
        q_len,
        sliding_window,
        decode_block_sizes=decode_block_sizes,
        poison_pre_window_nan=True,
        seed=seed,
    )


def _assert_matches(out: np.ndarray, ref: np.ndarray, kv_dtype: jnp.dtype) -> None:
    assert np.isfinite(out).all(), "NaN from pre-window KV reached the output"
    atol = 3e-2 if kv_dtype == jnp.bfloat16 else 6e-2
    np.testing.assert_allclose(out, ref, atol=atol, rtol=2e-2)


@pytest.mark.parametrize("q_len,distribution", [(1, (1, 1, 1)), (16, (0, 0, 1))])
def test_swa_intra_page_dma_trim_ignores_poisoned_pre_window_tokens(
    q_len, distribution
):
    """Pre-window tokens inside the boundary KV page must not affect output."""
    rng = np.random.default_rng(123)
    kv_len, sliding_window = 460, 256
    sw_start = _sw_start(kv_len, q_len, sliding_window)
    assert _PAGE < sw_start < 2 * _PAGE

    base_k = (rng.standard_normal((kv_len, _NUM_KV_HEADS, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )
    base_v = (rng.standard_normal((kv_len, _NUM_KV_HEADS, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )
    query = (rng.standard_normal((q_len, _NUM_Q_HEADS, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )

    clean_k, clean_v = base_k.copy(), base_v.copy()
    clean_k[:sw_start] = 0.0
    clean_v[:sw_start] = 0.0

    poison_k, poison_v = base_k.copy(), base_v.copy()
    poison_k[:sw_start] = 64.0
    poison_v[:sw_start] = -64.0

    control_k, control_v = clean_k.copy(), clean_v.copy()
    control_k[sw_start : sw_start + 8] = 64.0
    control_v[sw_start : sw_start + 8] = -64.0

    page_table = np.arange(8, dtype=np.int32).reshape(1, 8)
    out_clean, ref_clean = _seed_and_run_step(
        jnp.bfloat16,
        [clean_k],
        [clean_v],
        query,
        (kv_len,),
        q_len,
        sliding_window,
        page_table=page_table,
        total_pages=8,
        distribution=distribution,
    )
    out_poison, _ = _seed_and_run_step(
        jnp.bfloat16,
        [poison_k],
        [poison_v],
        query,
        (kv_len,),
        q_len,
        sliding_window,
        page_table=page_table,
        total_pages=8,
        distribution=distribution,
    )
    out_control, _ = _seed_and_run_step(
        jnp.bfloat16,
        [control_k],
        [control_v],
        query,
        (kv_len,),
        q_len,
        sliding_window,
        page_table=page_table,
        total_pages=8,
        distribution=distribution,
    )

    _assert_matches(out_clean, ref_clean, jnp.bfloat16)
    np.testing.assert_array_equal(out_clean, out_poison)
    assert not np.array_equal(out_clean, out_control)


@pytest.mark.parametrize("mask_invalid_v", [False, True])
def test_swa_fully_skipped_page_on_last_physical_page(mask_invalid_v):
    """A wholly pre-window page mapped to the last HBM page must DMA nothing."""
    rng = np.random.default_rng(11)
    total_pages = 8
    kv_len, sliding_window = 600, 128
    # Map logical page 2 [256, 384) (wholly before sw_start=472) onto the last
    # physical page so an untrimmed offset would point past the HBM pool.
    page_table = np.array([[0, 1, total_pages - 1, 2, 3, 4, 5, 6]], dtype=np.int32)
    hist_k = (rng.standard_normal((kv_len, _NUM_KV_HEADS, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )
    hist_v = (rng.standard_normal((kv_len, _NUM_KV_HEADS, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )
    query = (rng.standard_normal((1, _NUM_Q_HEADS, _HEAD_DIM)) * 0.5).astype(np.float32)

    out, ref = _seed_and_run_step(
        jnp.bfloat16,
        [hist_k],
        [hist_v],
        query,
        (kv_len,),
        1,
        sliding_window,
        page_table=page_table,
        total_pages=total_pages,
        mask_invalid_v=mask_invalid_v,
        poison_pre_window_nan=True,
    )
    _assert_matches(out, ref, jnp.bfloat16)


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_swa_trim_skips_nan_pre_window_history(kv_dtype):
    """NaN history before sw_start must not reach PV (fails without the trim).

    W=256, bkv_sz=256, page_size=128, kv_len=973: sw_start=717 is mid-page in
    page 1 of the KV block [512, 768), so page 0 of the block is skipped
    entirely and page 1 is trimmed by 77 tokens.
    """
    out, ref = _run_poisoned_decode(kv_dtype, (973,), 1, 256)
    _assert_matches(out, ref, kv_dtype)


# kv_len for decode_query_size=1, W=256, bkv_sz=256, page_size=128, so
# sw_start = kv_len - 256. Positions relative to the KV block [512, 768).
_SW_START_CASES = {
    "page0_offset0": 768,  # sw_start=512: block- and page-aligned, no trim
    "page0_mid": 805,  # sw_start=549: mid-page in page 0
    "page1_boundary": 896,  # sw_start=640: page 0 skipped, page 1 whole
    "page0_last_token": 895,  # sw_start=639: last token of page 0
    "page1_mid": 973,  # sw_start=717: page 0 skipped, page 1 trimmed
    "clamped": 100,  # sw_start<0: clamped to 0, nothing to trim
}


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("case", list(_SW_START_CASES))
def test_swa_trim_sw_start_positions(kv_dtype, case):
    out, ref = _run_poisoned_decode(kv_dtype, (_SW_START_CASES[case],), 1, 256)
    _assert_matches(out, ref, kv_dtype)


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
def test_swa_trim_batch_with_different_offsets(kv_dtype):
    """Several sequences in one decode batch, each with its own sw_start."""
    kv_lens = tuple(_SW_START_CASES.values())
    out, ref = _run_poisoned_decode(kv_dtype, kv_lens, 1, 256)
    _assert_matches(out, ref, kv_dtype)


@pytest.mark.parametrize("kv_dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("q_len", [2, 4])
def test_swa_trim_decode_query_size_gt_1(kv_dtype, q_len):
    """decode_query_size > 1: later queries have later windows than sw_start."""
    kv_lens = (805, 973, 896, 300)
    out, ref = _run_poisoned_decode(
        kv_dtype, kv_lens, q_len, 256, decode_block_sizes=None
    )
    _assert_matches(out, ref, kv_dtype)


def test_swa_trim_leftover_vmem_from_other_sequences():
    """Trimmed VMEM slots that held another sequence's KV must stay masked.

    24 decode sequences with batch_size=8 cycle through the three VMEM buffer
    slots, so trimmed slots of later steps still hold KV from earlier
    sequences in the same call. Even sequences use large-magnitude KV so a
    leak would be far outside tolerance.
    """
    kv_lens = tuple(
        (768, 805, 896, 895, 973, 1000)[i % 6] if i % 2 else 1024 + 37 * (i % 4)
        for i in range(24)
    )
    loud = tuple(range(0, 24, 2))
    out, ref = _run_poisoned_decode(jnp.bfloat16, kv_lens, 1, 256, loud_seqs=loud)
    quiet = np.arange(1, 24, 2)
    _assert_matches(out[quiet], ref[quiet], jnp.bfloat16)
    assert np.isfinite(out).all()


@pytest.mark.parametrize("sliding_window", [None, 1024])
def test_page_size_larger_than_mxu_auto_block_sizes(sliding_window):
    """page_size (512) > mxu_column_size: auto block sizes stay page-aligned."""
    rng = np.random.default_rng(5)
    kv_lens = (1500, 700, 2100)
    ks = [
        (rng.standard_normal((n, _NUM_KV_HEADS, _HEAD_DIM)) * 0.5).astype(np.float32)
        for n in kv_lens
    ]
    vs = [
        (rng.standard_normal((n, _NUM_KV_HEADS, _HEAD_DIM)) * 0.5).astype(np.float32)
        for n in kv_lens
    ]
    q = (rng.standard_normal((len(kv_lens), _NUM_Q_HEADS, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )
    out, ref = _seed_and_run_step(
        jnp.bfloat16,
        ks,
        vs,
        q,
        kv_lens,
        1,
        sliding_window,
        page_size=512,
        decode_block_sizes=None,
        prefill_block_sizes=None,
        poison_pre_window_nan=sliding_window is not None,
        seed=5,
    )
    _assert_matches(out, ref, jnp.bfloat16)
