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

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as kernels,
)


def _reference_packed_mla_attention(
    query_bf16: jnp.ndarray,
    key_bf16: jnp.ndarray,
    value_bf16: jnp.ndarray,
    segment_specs: tuple[tuple[int, int, int, int], ...],
    arena_k: jnp.ndarray | None,
    arena_v: jnp.ndarray | None,
) -> np.ndarray:
    q_np = np.asarray(query_bf16, dtype=np.float32) * np.log(2.0)
    k_np = np.asarray(key_bf16, dtype=np.float32)
    v_np = np.asarray(value_bf16, dtype=np.float32)
    heads, tokens, _ = q_np.shape
    ref = np.zeros((heads, tokens, 128), dtype=np.float32)
    ak_np = None if arena_k is None else np.asarray(arena_k[0], dtype=np.float32)
    av_np = (
        None
        if arena_v is None
        else np.asarray(arena_v[0], dtype=np.float32).transpose(0, 2, 1)
    )
    for offset, seg_len, arena_base, cached_len in segment_specs:
        for h_idx in range(heads):
            q_seg = q_np[h_idx, offset : offset + seg_len]
            k_win = k_np[h_idx, offset : offset + seg_len]
            v_win = v_np[h_idx, offset : offset + seg_len]
            if ak_np is not None and av_np is not None and cached_len > 0:
                k_all = np.concatenate(
                    [ak_np[h_idx, arena_base : arena_base + cached_len], k_win], axis=0
                )
                v_all = np.concatenate(
                    [av_np[h_idx, arena_base : arena_base + cached_len], v_win], axis=0
                )
            else:
                k_all = k_win
                v_all = v_win
            logits = q_seg @ k_all.T
            q_rows = np.arange(seg_len)[:, None]
            k_cols = np.arange(seg_len)[None, :]
            mask = np.concatenate(
                [
                    np.ones((seg_len, cached_len), dtype=bool),
                    k_cols <= q_rows,
                ],
                axis=1,
            )
            logits = np.where(mask, logits, -1e30)
            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)
            ref[h_idx, offset : offset + seg_len] = probs @ v_all
    return ref


def test_should_match_fp32_oracle_when_packed_mla_flash_runs_without_prefix():
    rng = np.random.default_rng(2026112301)
    tokens, heads, live_tokens = 512, 3, 448
    q_np = rng.standard_normal((heads, tokens, 256), dtype=np.float32) * 0.15
    k_np = rng.standard_normal((heads, tokens, 256), dtype=np.float32) * 0.3
    v_np = rng.standard_normal((heads, tokens, 128), dtype=np.float32) * 0.3
    k_np[:, live_tokens:] = 999.0
    v_np[:, live_tokens:] = 999.0

    query = jnp.asarray(q_np, dtype=jnp.bfloat16)
    key = jnp.asarray(k_np, dtype=jnp.bfloat16)
    value = jnp.asarray(v_np, dtype=jnp.bfloat16)
    segments = jnp.asarray([0, 256, 256, 192, 0, 0], dtype=jnp.int32)
    length = jnp.asarray([live_tokens], dtype=jnp.int32)

    out = kernels.mla_flash_prefill_packed(
        query,
        key,
        value,
        length,
        segments,
        count=2,
        interpret=True,
    )
    ref = _reference_packed_mla_attention(
        query,
        key,
        value,
        ((0, 256, 0, 0), (256, 192, 0, 0)),
        None,
        None,
    )
    out_np = np.asarray(out, dtype=np.float32)
    assert np.all(np.isfinite(out_np))
    np.testing.assert_array_equal(out_np[:, live_tokens:, :], 0.0)
    rel_l2 = float(np.linalg.norm(out_np - ref) / (np.linalg.norm(ref) + 1e-8))
    assert rel_l2 < 0.01


def test_should_match_fp32_oracle_when_mla_flash_streams_4k_window_and_fp8_prefix():
    rng = np.random.default_rng(2026112302)
    tokens, heads, live_tokens = 4096, 3, 3840
    q_np = rng.standard_normal((heads, tokens, 256), dtype=np.float32) * 0.15
    k_np = rng.standard_normal((heads, tokens, 256), dtype=np.float32) * 0.3
    v_np = rng.standard_normal((heads, tokens, 128), dtype=np.float32) * 0.3
    k_np[:, live_tokens:] = 777.0
    v_np[:, live_tokens:] = 777.0

    ak_np = rng.standard_normal((1, heads, 6144, 256), dtype=np.float32) * 0.2
    av_np = rng.standard_normal((1, heads, 128, 6144), dtype=np.float32) * 0.2
    ak_np[:, :, 2304:2560, :] = 400.0
    av_np[:, :, :, 2304:2560] = 400.0
    ak_np[:, :, 5888:, :] = 400.0
    av_np[:, :, :, 5888:] = 400.0

    query = jnp.asarray(q_np, dtype=jnp.bfloat16)
    key = jnp.asarray(k_np, dtype=jnp.bfloat16)
    value = jnp.asarray(v_np, dtype=jnp.bfloat16)
    arena_k = jnp.asarray(ak_np, dtype=jnp.float8_e4m3fn)
    arena_v = jnp.asarray(av_np, dtype=jnp.float8_e4m3fn)
    segments = jnp.asarray(
        [0, 2560, 2560, 1280, 0, 0, 0, 2560, 2304, 3328], dtype=jnp.int32
    )
    length = jnp.asarray([live_tokens], dtype=jnp.int32)

    out = kernels.mla_flash_prefill_packed(
        query,
        key,
        value,
        length,
        segments,
        count=2,
        arena=(arena_k, arena_v),
        layer=0,
        interpret=True,
    )
    ref = _reference_packed_mla_attention(
        query,
        key,
        value,
        ((0, 2560, 0, 2304), (2560, 1280, 2560, 3328)),
        arena_k,
        arena_v,
    )
    out_np = np.asarray(out, dtype=np.float32)
    assert np.all(np.isfinite(out_np))
    np.testing.assert_array_equal(out_np[:, live_tokens:, :], 0.0)
    rel_l2 = float(np.linalg.norm(out_np - ref) / (np.linalg.norm(ref) + 1e-8))
    assert rel_l2 < 0.02
