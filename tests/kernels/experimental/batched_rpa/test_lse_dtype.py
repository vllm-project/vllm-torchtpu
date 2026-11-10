# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""lse_dtype=float32 returns an LSE that is accurate enough to merge with.

The default LSE is stored in the output dtype (bf16): at magnitudes of 8-32
its spacing is 0.0625-0.125, which mis-weights an LSE merge with partial
results from another kernel by several percent. With lse_dtype=float32 the
kernel forms m + log(l) in f32 and stores it unrounded.
"""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

PAGE = 128
NUM_KV_HEADS, Q_PER_KV, HEAD_DIM = 2, 16, 256


class LseDtypeTest(jtu.JaxTestCase):
    # SEQ_ALONG_LANE is the production (HND) layout. HEAD_ALONG_SUBLANE is left
    # out: its default decode tiles for this geometry exceed VMEM in the cache
    # fill (RPAd-snh-p128-b8-q1-k2048 needs 65.86M of 63.94M, j-685a609f).
    # KVLayout is a StrEnum: wrap it, or absl treats it as a bare string.
    @parameterized.parameters((configs.KVLayout.SEQ_ALONG_LANE,))
    def test_f32_lse_tracks_reference(self, kv_layout):
        if not jtu.is_device_tpu_at_least(version=4):
            self.skipTest("Expect TPUv4+")
        rng = np.random.default_rng(3)
        history = (2048, 1000)
        q_lens = (64, 48)
        num_q = NUM_KV_HEADS * Q_PER_KV
        pages_per_seq = 24
        num_pages = len(history) * pages_per_seq
        k_hist = [
            np.asarray(jnp.asarray(rng.normal(0, 1, (h, NUM_KV_HEADS, HEAD_DIM)), jnp.float8_e4m3fn).astype(jnp.float32))
            for h in history
        ]
        v_hist = [
            np.asarray(jnp.asarray(rng.normal(0, 1, (h, NUM_KV_HEADS, HEAD_DIM)), jnp.float8_e4m3fn).astype(jnp.float32))
            for h in history
        ]
        # Large logits so the LSE sits around 10-30.
        q = np.asarray(
            jnp.asarray(rng.normal(0, 3, (sum(q_lens), num_q, HEAD_DIM)), jnp.bfloat16).astype(jnp.float32)
        )
        sm_scale = HEAD_DIM**-0.5
        page_indices = np.arange(num_pages, dtype=np.int32)

        # Fill the cache through the kernel itself: one FULL pass per sequence
        # writes its history as new tokens.
        shape = wrapper.get_kv_cache_shape(
            num_pages, PAGE, NUM_KV_HEADS, HEAD_DIM, jnp.float8_e4m3fn, kv_layout=kv_layout
        )
        cache = jnp.zeros(shape, jnp.float8_e4m3fn)
        for i, h in enumerate(history):
            kv_lens = np.zeros(len(history), np.int32)
            kv_lens[i] = h
            cu = np.zeros(len(history) + 1, np.int32)
            cu[i + 1 :] = h
            _, cache = wrapper.ragged_paged_attention(
                jnp.zeros((h, num_q, HEAD_DIM), jnp.bfloat16),
                jnp.asarray(k_hist[i], jnp.float8_e4m3fn),  # RPA takes K/V in the cache dtype
                jnp.asarray(v_hist[i], jnp.float8_e4m3fn),
                cache,
                jnp.asarray(kv_lens),
                jnp.asarray(page_indices),
                jnp.asarray(cu),
                jnp.array([0, 0, len(history)], jnp.int32),
                sm_scale=sm_scale,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=kv_layout,
            )

        kv_lens = jnp.asarray(np.asarray(history) + np.asarray(q_lens), jnp.int32)
        cu = jnp.asarray(np.cumsum((0, *q_lens)), jnp.int32)
        zeros_kv = jnp.zeros((sum(q_lens), NUM_KV_HEADS, HEAD_DIM), jnp.float8_e4m3fn)

        def run(lse_dtype):
            _, _, lse = wrapper.ragged_paged_attention(
                jnp.asarray(q, jnp.bfloat16),
                zeros_kv,
                zeros_kv,
                jnp.copy(cache),  # the wrapper donates kv_cache
                kv_lens,
                jnp.asarray(page_indices),
                cu,
                jnp.array([0, 0, len(history)], jnp.int32),
                sm_scale=sm_scale,
                k_scale=1.0,
                v_scale=1.0,
                kv_layout=kv_layout,
                attention_scope=configs.AttentionScope.CACHE_ONLY,
                return_lse=True,
                lse_dtype=lse_dtype,
                skip_kv_update=True,
            )
            return np.asarray(lse.astype(jnp.float32))

        expected = np.zeros((sum(q_lens), num_q))
        row = 0
        for i, ql in enumerate(q_lens):
            kk = np.repeat(k_hist[i], Q_PER_KV, axis=1)
            for t in range(ql):
                s = np.einsum("hd,nhd->hn", q[row], kk) * sm_scale
                m = s.max(-1)
                expected[row] = m + np.log(np.exp(s - m[:, None]).sum(-1))
                row += 1

        lse_f32 = run(jnp.float32)
        lse_default = run(None)
        err_f32 = np.abs(lse_f32 - expected)
        err_default = np.abs(lse_default - expected)
        print(
            f"{kv_layout}: lse range [{expected.min():.2f}, {expected.max():.2f}] "
            f"f32 max/mean {err_f32.max():.4f}/{err_f32.mean():.5f} "
            f"default max/mean {err_default.max():.4f}/{err_default.mean():.5f}"
        )
        self.assertLess(err_f32.max(), 0.02)
        self.assertLess(err_f32.mean(), 0.5 * err_default.mean())


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
