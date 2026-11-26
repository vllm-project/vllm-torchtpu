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
"""TPU Pallas kernel parity tests for Subspace-RPA across layouts and regimes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

pytestmark = pytest.mark.tpu_test

_HEAD_DIM = 512
_ROT_DIM = 128
_PAGE_SIZE = 128
_K_SCALE = 0.75
_V_SCALE = 1.25
_DECODE_BLOCKS = configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8)
_PREFILL_BLOCKS = configs.BlockSizes(bq_sz=256, bq_c_sz=32, bkv_sz=256, batch_size=2)


def _quantize_fp8(x: np.ndarray, scale: float) -> jax.Array:
    finfo = jnp.finfo(jnp.float8_e4m3fn)
    scaled = jnp.clip(
        jnp.asarray(x, dtype=jnp.float32) / scale,
        float(finfo.min),
        float(finfo.max),
    )
    return scaled.astype(jnp.float8_e4m3fn)


def _reference_subspace_attention(
    queries: np.ndarray,
    full_k_rot: list[np.ndarray],
    full_v: list[np.ndarray],
    q_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    kv_layout: configs.KVLayout,
    sm_scale: float,
) -> np.ndarray:
    """Independent causal GQA reference reconstructing 512-dim K from (K_rot, V)."""
    num_q_heads = queries.shape[1]
    num_kv_heads = full_v[0].shape[1]
    q_per_kv = num_q_heads // num_kv_heads
    outputs: list[np.ndarray] = []
    q_offset = 0

    for seq_idx, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        q_seq = jnp.asarray(queries[q_offset : q_offset + q_len], jnp.bfloat16).astype(
            jnp.float32
        )
        q_offset += q_len

        v_q = _quantize_fp8(full_v[seq_idx][:kv_len], _V_SCALE)
        v_deq = v_q.astype(jnp.float32) * _V_SCALE

        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            kr_q = _quantize_fp8(full_k_rot[seq_idx][:kv_len], _K_SCALE)
            kr_deq = kr_q.astype(jnp.float32) * _K_SCALE
        else:
            kr_deq = jnp.asarray(full_k_rot[seq_idx][:kv_len], jnp.bfloat16).astype(
                jnp.float32
            )

        k_full = jnp.concatenate([kr_deq, v_deq[..., _ROT_DIM:]], axis=-1)
        k_rep = jnp.repeat(k_full, q_per_kv, axis=1)
        v_rep = jnp.repeat(v_deq, q_per_kv, axis=1)

        scores = jnp.einsum("thd,shd->hts", q_seq, k_rep) * sm_scale
        q_pos = (kv_len - q_len) + jnp.arange(q_len)[:, None]
        k_pos = jnp.arange(kv_len)[None, :]
        mask = k_pos <= q_pos
        scores = jnp.where(mask[None, :, :], scores, -1e30)
        probs = jax.nn.softmax(scores, axis=-1)
        out_seq = jnp.einsum("hts,shd->thd", probs, v_rep).astype(jnp.bfloat16)
        outputs.append(np.asarray(out_seq.astype(jnp.float32)))

    return np.concatenate(outputs, axis=0)


def _pack_expected_page_token(
    k_rot_tok: np.ndarray,
    v_tok: np.ndarray,
    kv_layout: configs.KVLayout,
) -> np.ndarray:
    """Pack a single token's expected KV bytes independently of the kernel."""
    v_fp8 = _quantize_fp8(v_tok, _V_SCALE)  # [H_kv, 512]
    num_kv_heads = v_tok.shape[0]
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        k_fp8 = _quantize_fp8(k_rot_tok, _K_SCALE)  # [H_kv, 128]
        cat = jnp.concatenate([k_fp8, v_fp8], axis=-1)  # [H_kv, 640]
        return np.asarray(
            jax.lax.bitcast_convert_type(cat.reshape(num_kv_heads, 160, 4), jnp.uint8)
        )
    # HEAD_ALONG_SUBLANE: [v0, v1, k_rot_lo (BF16 lowB), k_rot_hi (BF16 highB)]
    v0 = v_fp8[:, None, :256]
    v1 = v_fp8[:, None, 256:512]
    k_bf16 = jnp.pad(
        jnp.asarray(k_rot_tok, jnp.bfloat16), ((0, 0), (0, 256 - _ROT_DIM))
    )
    k_bytes = jax.lax.bitcast_convert_type(k_bf16, jnp.float8_e4m3fn).swapaxes(-1, -2)
    packed = jnp.concatenate([v0, v1, k_bytes], axis=1)  # [H_kv, 4, 256]
    return np.asarray(jax.lax.bitcast_convert_type(packed, jnp.uint8))


def _extract_cache_token_bytes(
    cache: jax.Array,
    page_id: int,
    offset: int,
    kv_layout: configs.KVLayout,
) -> np.ndarray:
    cache_u8 = jax.lax.bitcast_convert_type(cache, jnp.uint8)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        return np.asarray(cache_u8[page_id, :, :, :, offset])
    return np.asarray(cache_u8[page_id, offset, :, :, :])


def _prepare_step_inputs(
    queries: np.ndarray,
    k_rot: np.ndarray,
    values: np.ndarray,
    kv_layout: configs.KVLayout,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    q_jax = jnp.asarray(queries, jnp.bfloat16)
    v_jax = _quantize_fp8(values, _V_SCALE)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        k_jax = _quantize_fp8(k_rot, _K_SCALE)
    else:
        k_jax = jnp.asarray(k_rot, jnp.bfloat16)
    return q_jax, k_jax, v_jax


@pytest.mark.parametrize(
    "kv_layout",
    [configs.KVLayout.HEAD_ALONG_SUBLANE, configs.KVLayout.SEQ_ALONG_LANE],
)
@pytest.mark.parametrize(
    "num_kv_heads",
    [
        1,
        # num_kv_heads=2 hits has_bank_conflicts(2) -> bkv_stride=3 != kv_hbm_stride=2.
        2,
    ],
)
def test_subspace_rpa_prefill_decode_and_mixed_parity(
    kv_layout: configs.KVLayout,
    num_kv_heads: int,
) -> None:
    """Verify Subspace-RPA output and cache bytes on prefill, decode, and mixed."""
    rng = np.random.default_rng(17 + num_kv_heads)
    num_q_heads = num_kv_heads * 4
    sm_scale = _HEAD_DIM**-0.5
    pages_per_seq = 3
    num_seqs = 2
    # Allocate 1 extra canary page at the end to verify untouched pages stay intact.
    total_pages = num_seqs * pages_per_seq + 1
    canary_page = total_pages - 1
    page_indices = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)

    max_len = pages_per_seq * _PAGE_SIZE
    full_k_rot = [
        (rng.standard_normal((max_len, num_kv_heads, _ROT_DIM)) * 0.6).astype(
            np.float32
        )
        for _ in range(num_seqs)
    ]
    full_v = [
        (rng.standard_normal((max_len, num_kv_heads, _HEAD_DIM)) * 0.6).astype(
            np.float32
        )
        for _ in range(num_seqs)
    ]

    cache_shape = wrapper.get_kv_cache_shape(
        total_pages,
        _PAGE_SIZE,
        num_kv_heads,
        _HEAD_DIM,
        jnp.float8_e4m3fn,
        kv_layout=kv_layout,
        subspace_k_rot_dim=_ROT_DIM,
    )
    canary_bytes = rng.integers(1, 100, size=cache_shape[1:], dtype=np.uint8)
    # Fill unused cache slots with finite non-zero bytes (0x38 = 1.0 in fp8_e4m3fn)
    # so out-of-bounds reads or missing causal masks cannot hide behind zeros.
    init_u8 = np.full(cache_shape, 0x38, dtype=np.uint8)
    init_u8[canary_page] = canary_bytes
    cache = jax.lax.bitcast_convert_type(jnp.asarray(init_u8), jnp.float8_e4m3fn)

    # Step 1: Fresh multi-sequence prefill spanning multiple pages with partial tails.
    prefill_lens = (160, 140)
    q_prefill = (
        rng.standard_normal((sum(prefill_lens), num_q_heads, _HEAD_DIM)) * 0.5
    ).astype(np.float32)
    k_prefill = np.concatenate(
        [full_k_rot[i][: prefill_lens[i]] for i in range(num_seqs)], axis=0
    )
    v_prefill = np.concatenate(
        [full_v[i][: prefill_lens[i]] for i in range(num_seqs)], axis=0
    )
    q_jax, k_jax, v_jax = _prepare_step_inputs(
        q_prefill, k_prefill, v_prefill, kv_layout
    )
    out_prefill, cache = wrapper.ragged_paged_attention(
        q_jax,
        k_jax,
        v_jax,
        cache,
        jnp.asarray(prefill_lens, jnp.int32),
        page_indices,
        jnp.array([0, prefill_lens[0], sum(prefill_lens)], jnp.int32),
        jnp.array([0, 0, num_seqs], jnp.int32),
        sm_scale=sm_scale,
        k_scale=_K_SCALE,
        v_scale=_V_SCALE,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        kv_layout=kv_layout,
    )
    ref_prefill = _reference_subspace_attention(
        q_prefill,
        full_k_rot,
        full_v,
        prefill_lens,
        prefill_lens,
        kv_layout,
        sm_scale,
    )
    np.testing.assert_allclose(
        np.asarray(out_prefill.astype(jnp.float32)),
        ref_prefill,
        atol=2.5e-2,
        rtol=2e-2,
    )

    # Step 2: Pure decode over multi-page cached history.
    decode_kv_lens = (prefill_lens[0] + 1, prefill_lens[1] + 1)
    q_decode = (rng.standard_normal((num_seqs, num_q_heads, _HEAD_DIM)) * 0.5).astype(
        np.float32
    )
    k_decode = np.stack(
        [full_k_rot[i][prefill_lens[i]] for i in range(num_seqs)], axis=0
    )
    v_decode = np.stack([full_v[i][prefill_lens[i]] for i in range(num_seqs)], axis=0)
    q_jax, k_jax, v_jax = _prepare_step_inputs(q_decode, k_decode, v_decode, kv_layout)
    out_decode, cache = wrapper.ragged_paged_attention(
        q_jax,
        k_jax,
        v_jax,
        cache,
        jnp.asarray(decode_kv_lens, jnp.int32),
        page_indices,
        jnp.array([0, 1, 2], jnp.int32),
        jnp.array([num_seqs, num_seqs, num_seqs], jnp.int32),
        sm_scale=sm_scale,
        k_scale=_K_SCALE,
        v_scale=_V_SCALE,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        kv_layout=kv_layout,
    )
    ref_decode = _reference_subspace_attention(
        q_decode,
        full_k_rot,
        full_v,
        (1, 1),
        decode_kv_lens,
        kv_layout,
        sm_scale,
    )
    np.testing.assert_allclose(
        np.asarray(out_decode.astype(jnp.float32)),
        ref_decode,
        atol=2.5e-2,
        rtol=2e-2,
    )

    # Step 3: Mixed batch (seq 0 decode q_len=1, seq 1 chunked prefill q_len=37
    # crossing into a third page).
    mixed_q_lens = (1, 37)
    mixed_kv_lens = (
        decode_kv_lens[0] + mixed_q_lens[0],
        decode_kv_lens[1] + mixed_q_lens[1],
    )
    q_mixed = (
        rng.standard_normal((sum(mixed_q_lens), num_q_heads, _HEAD_DIM)) * 0.5
    ).astype(np.float32)
    k_mixed = np.concatenate(
        [
            full_k_rot[0][decode_kv_lens[0] : mixed_kv_lens[0]],
            full_k_rot[1][decode_kv_lens[1] : mixed_kv_lens[1]],
        ],
        axis=0,
    )
    v_mixed = np.concatenate(
        [
            full_v[0][decode_kv_lens[0] : mixed_kv_lens[0]],
            full_v[1][decode_kv_lens[1] : mixed_kv_lens[1]],
        ],
        axis=0,
    )
    q_jax, k_jax, v_jax = _prepare_step_inputs(q_mixed, k_mixed, v_mixed, kv_layout)
    out_mixed, cache = wrapper.ragged_paged_attention(
        q_jax,
        k_jax,
        v_jax,
        cache,
        jnp.asarray(mixed_kv_lens, jnp.int32),
        page_indices,
        jnp.array([0, mixed_q_lens[0], sum(mixed_q_lens)], jnp.int32),
        jnp.array([1, 1, num_seqs], jnp.int32),
        sm_scale=sm_scale,
        k_scale=_K_SCALE,
        v_scale=_V_SCALE,
        decode_block_sizes=_DECODE_BLOCKS,
        prefill_block_sizes=_PREFILL_BLOCKS,
        kv_layout=kv_layout,
    )
    ref_mixed = _reference_subspace_attention(
        q_mixed,
        full_k_rot,
        full_v,
        mixed_q_lens,
        mixed_kv_lens,
        kv_layout,
        sm_scale,
    )
    np.testing.assert_allclose(
        np.asarray(out_mixed.astype(jnp.float32)),
        ref_mixed,
        atol=2.5e-2,
        rtol=2e-2,
    )

    # Verify stored KV cache bytes at boundary tokens (first page, page boundary,
    # decode append, and chunked-prefill tail) and verify canary page is untouched.
    for seq_idx, kv_len in enumerate(mixed_kv_lens):
        for tok_pos in (0, _PAGE_SIZE - 1, _PAGE_SIZE, kv_len - 1):
            page_id = seq_idx * pages_per_seq + (tok_pos // _PAGE_SIZE)
            offset = tok_pos % _PAGE_SIZE
            got_bytes = _extract_cache_token_bytes(cache, page_id, offset, kv_layout)
            want_bytes = _pack_expected_page_token(
                full_k_rot[seq_idx][tok_pos],
                full_v[seq_idx][tok_pos],
                kv_layout,
            )
            np.testing.assert_array_equal(got_bytes, want_bytes)

    after_canary = np.asarray(
        jax.lax.bitcast_convert_type(cache, jnp.uint8)[canary_page]
    )
    np.testing.assert_array_equal(after_canary, canary_bytes)
