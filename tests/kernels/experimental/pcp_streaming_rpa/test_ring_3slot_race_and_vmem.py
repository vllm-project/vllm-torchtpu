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
"""3-slot PCP RPA at the Qwen3.5-397B PCP8 serving geometry vs float64."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.kernels.experimental.batched_rpa.configs import KVLayout
from vllm_torchtpu.kernels.experimental.batched_rpa.utils import get_dtype_packing
from vllm_torchtpu.kernels.experimental.batched_rpa.wrapper import get_kv_cache_shape
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import wrapper as pcp_wrapper
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.pcp_layout import (
    build_pcp_rank_major_token_order,
)

PCP_AXIS = pcp_wrapper.PCP_AXIS_NAME
TP_AXIS = pcp_wrapper.TP_AXIS_NAME
EPS_BF16 = 2.0**-7
U_BF16 = 2.0**-8
U_FP32 = 2.0**-24

# Qwen3.5-397B PCP8 TP1 serving geometry, as in test_pcp8_160k_chunk_benchmark.py.
QWEN35_PCP_SIZE = 8
QWEN35_NUM_Q_HEADS = 32
QWEN35_NUM_KV_HEADS = 2
QWEN35_HEAD_DIM = 256
QWEN35_PAGE_SIZE = 256
QWEN35_INTERLEAVE_SIZE = 256
# Non-unit FP8 scale so a missing or doubled dequant scale changes the output.
FP8_KV_SCALE = 0.5


def _require_tpu_mesh(mesh_shape: tuple[int, ...]) -> Mesh:
    if jax.default_backend() != "tpu":
        pytest.skip("3-slot PCP streaming RPA tests require TPU.")
    devices = jax.local_devices()
    if len(devices) < 8:
        pytest.skip("3-slot PCP streaming RPA tests require 8 TPU devices.")
    axis_names = (PCP_AXIS,) if len(mesh_shape) == 1 else (PCP_AXIS, TP_AXIS)
    return Mesh(np.asarray(devices[:8]).reshape(mesh_shape), axis_names)


def _derived_fp64_tolerances(
    ref64: np.ndarray,
    v_fp64: np.ndarray,
    *,
    num_kv_tiles: int,
) -> tuple[float, float]:
    """Derive (tol_rmse, tol_max_abs) from IEEE constants and reference norms."""
    ref_rms = float(np.sqrt(np.mean(ref64**2)))
    v_rms = float(np.sqrt(np.mean(v_fp64**2)))
    ref_abs_max = float(np.max(np.abs(ref64)))
    v_abs_max = float(np.max(np.abs(v_fp64)))
    tol_rmse = math.sqrt(2.0) * U_BF16 * ref_rms + num_kv_tiles * U_FP32 * v_rms
    tol_max_abs = (
        2.0 * EPS_BF16 * ref_abs_max
        + EPS_BF16 * v_abs_max
        + num_kv_tiles * (2.0 * U_FP32) * v_abs_max
    )
    return tol_rmse, tol_max_abs


def _naive_causal_attention_fp64(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    request_absolute_start: int,
    sm_scale: float,
) -> np.ndarray:
    q_f64 = np.asarray(q, dtype=np.float64)
    k_f64 = np.asarray(k, dtype=np.float64)
    v_f64 = np.asarray(v, dtype=np.float64)
    q_per_kv = q_f64.shape[1] // k_f64.shape[1]
    q_positions = request_absolute_start + np.arange(q_f64.shape[0])
    causal_mask = q_positions[:, None] >= np.arange(k_f64.shape[0])[None, :]
    out = np.empty_like(q_f64)
    for q_head in range(q_f64.shape[1]):
        kv_head = q_head // q_per_kv
        scores = (q_f64[:, q_head] @ k_f64[:, kv_head].T) * np.float64(sm_scale)
        scores = np.where(causal_mask, scores, -np.inf)
        scores -= np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= np.sum(probs, axis=-1, keepdims=True)
        out[:, q_head] = probs @ v_f64[:, kv_head]
    return out


def _pack_multi_request_kv_cache(
    k_requests: list[np.ndarray],
    v_requests: list[np.ndarray],
    *,
    pcp_size: int,
    tp_size: int,
    page_size: int,
    interleave_size: int,
    kv_dtype: jnp.dtype,
    kv_layout: KVLayout,
    k_scale: float | None,
    v_scale: float | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    virtual_page_size = pcp_size * page_size
    pages_per_req = [
        math.ceil(k_req.shape[0] / virtual_page_size) for k_req in k_requests
    ]
    req_page_starts = np.cumsum([0] + pages_per_req[:-1])
    total_local_pages = max(sum(pages_per_req), 1)
    max_pages = max(pages_per_req)
    block_tables = np.zeros((len(k_requests), max_pages), dtype=np.int32)

    total_kv_heads, head_dim = k_requests[0].shape[1], k_requests[0].shape[2]
    local_kv_heads = total_kv_heads // tp_size
    packing = get_dtype_packing(kv_dtype)
    lane_groups = math.ceil((2 * local_kv_heads) / packing)
    padded_heads = lane_groups * packing
    rank_blocks = []
    for pcp_rank in range(pcp_size):
        tp_shards = []
        for tp_rank in range(tp_size):
            h0, h1 = tp_rank * local_kv_heads, (tp_rank + 1) * local_kv_heads
            local_k = np.zeros(
                (total_local_pages * page_size, local_kv_heads, head_dim),
                dtype=np.float32,
            )
            local_v = np.zeros_like(local_k)
            for req_idx, (k_req, v_req) in enumerate(zip(k_requests, v_requests)):
                p_start = int(req_page_starts[req_idx])
                block_tables[req_idx, : pages_per_req[req_idx]] = p_start + np.arange(
                    pages_per_req[req_idx], dtype=np.int32
                )
                for pos in range(k_req.shape[0]):
                    vpage = pos // virtual_page_size
                    voff = pos % virtual_page_size
                    chunk_idx = voff // interleave_size
                    if chunk_idx % pcp_size != pcp_rank:
                        continue
                    local_offset = (
                        chunk_idx // pcp_size
                    ) * interleave_size + voff % interleave_size
                    row = (p_start + vpage) * page_size + local_offset
                    local_k[row] = k_req[pos, h0:h1] / (k_scale or 1.0)
                    local_v[row] = v_req[pos, h0:h1] / (v_scale or 1.0)
            kv_pair = jnp.concatenate(
                [
                    jnp.asarray(local_k, dtype=jnp.bfloat16),
                    jnp.asarray(local_v, dtype=jnp.bfloat16),
                ],
                axis=-1,
            ).reshape(total_local_pages, page_size, 2 * local_kv_heads, head_dim)
            if kv_layout == KVLayout.SEQ_ALONG_LANE:
                packed = kv_pair.reshape(
                    total_local_pages,
                    page_size,
                    2 * local_kv_heads,
                    head_dim // packing,
                    packing,
                ).transpose(0, 2, 3, 4, 1)
            else:
                if padded_heads > 2 * local_kv_heads:
                    kv_pair = jnp.pad(
                        kv_pair,
                        (
                            (0, 0),
                            (0, 0),
                            (0, padded_heads - 2 * local_kv_heads),
                            (0, 0),
                        ),
                    )
                packed = kv_pair.reshape(
                    total_local_pages, page_size, lane_groups, packing, head_dim
                )
            tp_shards.append(packed.astype(kv_dtype))
        concat_axis = 1 if kv_layout == KVLayout.SEQ_ALONG_LANE else 2
        rank_blocks.append(jnp.concatenate(tp_shards, axis=concat_axis))
    return jnp.concatenate(rank_blocks, axis=0), jnp.asarray(
        block_tables.reshape(-1), dtype=jnp.int32
    )


def test_multi_request_kv_cache_packing_and_fp64_reference():
    """Verify multi-request KV cache packing and float64 causal reference on host."""
    rng = np.random.default_rng(7)
    k_reqs = [
        rng.normal(size=(256, 4, 128)).astype(np.float32) * 0.1,
        rng.normal(size=(128, 4, 128)).astype(np.float32) * 0.1,
    ]
    v_reqs = [
        rng.normal(size=(256, 4, 128)).astype(np.float32) * 0.1,
        rng.normal(size=(128, 4, 128)).astype(np.float32) * 0.1,
    ]
    for tp_size in (1, 2):
        for kv_dtype, kv_layout, scale in (
            (jnp.bfloat16, KVLayout.HEAD_ALONG_SUBLANE, None),
            (jnp.float8_e4m3fn, KVLayout.SEQ_ALONG_LANE, 0.5),
        ):
            cache, page_indices = _pack_multi_request_kv_cache(
                k_reqs,
                v_reqs,
                pcp_size=4,
                tp_size=tp_size,
                page_size=32,
                interleave_size=32,
                kv_dtype=kv_dtype,
                kv_layout=kv_layout,
                k_scale=scale,
                v_scale=scale,
            )
            expected_shape = list(
                get_kv_cache_shape(12, 32, 4 // tp_size, 128, kv_dtype, kv_layout)
            )
            if tp_size > 1:
                axis = 1 if kv_layout == KVLayout.SEQ_ALONG_LANE else 2
                expected_shape[axis] *= tp_size
            assert cache.shape == tuple(expected_shape)
            np.testing.assert_array_equal(
                np.asarray(page_indices), np.asarray([0, 1, 2, 0], dtype=np.int32)
            )

    q_sample = rng.normal(size=(16, 8, 128)).astype(np.float32) * 0.1
    ref = _naive_causal_attention_fp64(
        q_sample, k_reqs[1][:32], v_reqs[1][:32], 16, 128**-0.5
    )
    wrong_ref = _naive_causal_attention_fp64(
        q_sample, k_reqs[1][:32], -v_reqs[1][:32], 16, 128**-0.5
    )
    tol_rmse, tol_max_abs = _derived_fp64_tolerances(
        ref, np.asarray(v_reqs[1][:32], dtype=np.float64), num_kv_tiles=1
    )
    assert ref.shape == (16, 8, 128)
    assert float(np.sqrt(np.mean((ref - wrong_ref) ** 2))) > 5.0 * tol_rmse
    assert float(np.max(np.abs(ref - wrong_ref))) > 5.0 * tol_max_abs


def _qwen35_fixture(kv_dtype: jnp.dtype) -> dict:
    """Two prefill requests at the Qwen3.5 PCP8 geometry with float64 refs.

    Request 0 has 3 virtual page groups of history (a ring slot is reused at
    round 2), request 1 has 1. Each request adds 2048 tokens, so every rank
    holds one full q block. Inputs are unit-variance.
    """
    q_lens = [2048, 2048]
    history_lens = [
        3 * QWEN35_PCP_SIZE * QWEN35_PAGE_SIZE,
        QWEN35_PCP_SIZE * QWEN35_PAGE_SIZE,
    ]
    kv_lens = [h + q for h, q in zip(history_lens, q_lens)]
    sm_scale = QWEN35_HEAD_DIM**-0.5
    scale = FP8_KV_SCALE if kv_dtype == jnp.float8_e4m3fn else None

    def _bf16(x):
        return np.asarray(
            jnp.asarray(x, dtype=jnp.bfloat16).astype(jnp.float32), dtype=np.float32
        )

    def _cache_values(x):
        if scale is None:
            return _bf16(x)
        return _bf16(jnp.asarray(x / scale, dtype=kv_dtype).astype(jnp.float32) * scale)

    rng = np.random.default_rng(42)
    q_reqs = [
        _bf16(rng.normal(size=(n, QWEN35_NUM_Q_HEADS, QWEN35_HEAD_DIM))) for n in q_lens
    ]
    kv_shape = (QWEN35_NUM_KV_HEADS, QWEN35_HEAD_DIM)
    k_reqs = [_cache_values(rng.normal(size=(n, *kv_shape))) for n in kv_lens]
    v_reqs = [_cache_values(rng.normal(size=(n, *kv_shape))) for n in kv_lens]
    expected = np.concatenate(
        [
            _naive_causal_attention_fp64(q, k, v, h, sm_scale)
            for q, k, v, h in zip(q_reqs, k_reqs, v_reqs, history_lens)
        ]
    )
    tol_rmse, tol_max_abs = _derived_fp64_tolerances(
        expected,
        np.concatenate(v_reqs).astype(np.float64),
        num_kv_tiles=math.ceil(max(kv_lens) / QWEN35_PAGE_SIZE),
    )

    # Negative control: request 0 reads virtual page group 2 in place of group
    # 1, which is the output of a ring slot overwritten before it is consumed.
    vpage = QWEN35_PCP_SIZE * QWEN35_PAGE_SIZE
    k_stale, v_stale = k_reqs[0].copy(), v_reqs[0].copy()
    k_stale[vpage : 2 * vpage] = k_reqs[0][2 * vpage : 3 * vpage]
    v_stale[vpage : 2 * vpage] = v_reqs[0][2 * vpage : 3 * vpage]
    stale = expected.copy()
    stale[: q_lens[0]] = _naive_causal_attention_fp64(
        q_reqs[0], k_stale, v_stale, history_lens[0], sm_scale
    )
    return {
        "q_lens": q_lens,
        "history_lens": history_lens,
        "kv_lens": kv_lens,
        "sm_scale": sm_scale,
        "scale": scale,
        "q_reqs": q_reqs,
        "k_reqs": k_reqs,
        "v_reqs": v_reqs,
        "expected": expected,
        "stale": stale,
        "tol_rmse": tol_rmse,
        "tol_max_abs": tol_max_abs,
    }


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def test_qwen35_stale_slot_control_exceeds_rmse_tolerance():
    """A stale ring slot must move the float64 output beyond the RMSE gate."""
    fx = _qwen35_fixture(jnp.float8_e4m3fn)
    assert _rms(fx["stale"] - fx["expected"]) > 5.0 * fx["tol_rmse"]


@pytest.mark.tpu_test
@pytest.mark.multichip
@pytest.mark.parametrize(
    ("kv_dtype", "kv_layout"),
    [
        (jnp.float8_e4m3fn, KVLayout.HEAD_ALONG_SUBLANE),
        (jnp.float8_e4m3fn, KVLayout.SEQ_ALONG_LANE),
        (jnp.bfloat16, KVLayout.HEAD_ALONG_SUBLANE),
    ],
)
def test_3slot_ring_matches_fp64_at_qwen35_pcp8(
    release_jax_backend, kv_dtype, kv_layout
):
    """Run the 3-slot kernel at the Qwen3.5 PCP8 geometry and compare to float64."""
    mesh = _require_tpu_mesh((QWEN35_PCP_SIZE,))
    q_block_size = PCP_STREAMING_RPA_LOCAL_COMPILE_TOKEN_MULTIPLE
    global_tokens = QWEN35_PCP_SIZE * q_block_size
    fx = _qwen35_fixture(kv_dtype)
    assert _rms(fx["stale"] - fx["expected"]) > 5.0 * fx["tol_rmse"]
    q_lens, history_lens, scale = fx["q_lens"], fx["history_lens"], fx["scale"]

    token_order, inverse_order = build_pcp_rank_major_token_order(
        q_lens,
        pcp_size=QWEN35_PCP_SIZE,
        interleave_size=QWEN35_INTERLEAVE_SIZE,
        padded_num_tokens=global_tokens,
        token_owner_start_offsets_per_req=[0, q_lens[0]],
    )
    valid = token_order >= 0
    q_cat = np.concatenate(fx["q_reqs"])
    # The wrapper casts new K/V straight to the cache dtype; the kernel applies
    # the scale on read, so new tokens are passed pre-divided like the cache.
    k_cur = np.concatenate(
        [k[h:] for k, h in zip(fx["k_reqs"], history_lens)]
    ) / np.float32(scale or 1.0)
    v_cur = np.concatenate(
        [v[h:] for v, h in zip(fx["v_reqs"], history_lens)]
    ) / np.float32(scale or 1.0)
    q_packed = np.zeros((global_tokens, *q_cat.shape[1:]), dtype=np.float32)
    k_packed = np.zeros((global_tokens, *k_cur.shape[1:]), dtype=np.float32)
    v_packed = np.zeros_like(k_packed)
    q_packed[valid] = q_cat[token_order[valid]]
    k_packed[valid] = k_cur[token_order[valid]]
    v_packed[valid] = v_cur[token_order[valid]]

    # Seed history pages only; the kernel writes the new tokens.
    k_hist, v_hist = [], []
    for k, v, h in zip(fx["k_reqs"], fx["v_reqs"], history_lens):
        k_h, v_h = np.zeros_like(k), np.zeros_like(v)
        k_h[:h], v_h[:h] = k[:h], v[:h]
        k_hist.append(k_h)
        v_hist.append(v_h)
    kv_cache, page_indices = _pack_multi_request_kv_cache(
        k_hist,
        v_hist,
        pcp_size=QWEN35_PCP_SIZE,
        tp_size=1,
        page_size=QWEN35_PAGE_SIZE,
        interleave_size=QWEN35_INTERLEAVE_SIZE,
        kv_dtype=kv_dtype,
        kv_layout=kv_layout,
        k_scale=scale,
        v_scale=scale,
    )

    output, _ = pcp_wrapper.sharded_pcp_ragged_paged_attention(
        mesh=mesh,
        q=jnp.asarray(q_packed, dtype=jnp.bfloat16),
        k=jnp.asarray(k_packed, dtype=jnp.bfloat16),
        v=jnp.asarray(v_packed, dtype=jnp.bfloat16),
        kv_cache=kv_cache,
        kv_lens=jnp.asarray(fx["kv_lens"], dtype=jnp.int32),
        page_indices=page_indices,
        cu_q_lens=jnp.asarray([0, q_lens[0], sum(q_lens)], dtype=jnp.int32),
        distribution=jnp.asarray([0, 0, len(q_lens)], dtype=jnp.int32),
        attention_sink=None,
        sm_scale=fx["sm_scale"],
        k_scale=scale,
        v_scale=scale,
        cp_kv_cache_interleave_size=QWEN35_INTERLEAVE_SIZE,
        q_block_size=q_block_size,
        kv_layout=kv_layout,
    )
    out = np.asarray(jax.device_get(output), dtype=np.float64)[inverse_order]

    assert np.all(np.isfinite(out))
    assert _rms(out - fx["expected"]) <= fx["tol_rmse"]
    assert float(np.max(np.abs(out - fx["expected"]))) <= fx["tol_max_abs"]
    assert _rms(out - fx["stale"]) > 5.0 * fx["tol_rmse"]
