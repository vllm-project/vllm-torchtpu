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
"""TurboQuant 4-bit latent cache through the GLM-5.2 sparse MLA kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    KVCacheLayout,
    KVCacheType,
    SparseMLAKVCacheSpec,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla
from vllm_torchtpu.layers.core import attention_interface
from vllm_torchtpu.layers.core import turboquant as tq

pytestmark = pytest.mark.tpu_test

NOPE_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256


def _lloyd_max(rng, dim, levels=16, samples=400_000, iters=60):
    x = rng.standard_normal(samples) / np.sqrt(dim)
    c = np.quantile(x, (np.arange(levels) + 0.5) / levels)
    for _ in range(iters):
        idx = np.searchsorted((c[:-1] + c[1:]) / 2, x)
        c = np.array([x[idx == i].mean() for i in range(levels)])
    return np.sort(c).astype(np.float32)


def _cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1, a.shape[-1])
    b = np.asarray(b, np.float32).reshape(-1, b.shape[-1])
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def _build(
    rng,
    *,
    query_tokens,
    topk,
    num_heads=8,
    cache_tokens=8192,
    k_scale=1.0,
    codebook="cubic",
):
    pages = cache_tokens // PAGE_SIZE
    tables = tq.make_mla_latent_tables(
        None if codebook == "cubic" else _lloyd_max(rng, NOPE_DIM)
    )
    outliers = 1 + 3 * (rng.random(NOPE_DIM) < 0.02)
    latent = jnp.asarray(
        rng.standard_normal((cache_tokens, NOPE_DIM)) * outliers, jnp.bfloat16
    )
    k_pe = jnp.asarray(
        rng.standard_normal((cache_tokens, ROPE_DIM)) / k_scale, jnp.float32
    ).astype(jnp.float8_e4m3fn)

    nope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.NOPE,
        KVCacheLayout.SPARSECORE,
        pages,
        PAGE_SIZE,
        tq.MLA_TQ_NOPE_BYTES,
    )
    rope_spec = SparseMLAKVCacheSpec.create(
        KVCacheType.ROPE, KVCacheLayout.SPARSECORE, pages, PAGE_SIZE, ROPE_DIM
    )
    assert nope_spec.shape == (pages, PAGE_SIZE // 2, 128), nope_spec.shape
    page_table = jnp.asarray(rng.permutation(pages).astype(np.int32))
    codes, rope_row = attention_interface._tq_pack_sparse_mla_latent(
        latent, k_pe, tables
    )
    args = (
        jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
        jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
        codes,
        rope_row,
        jnp.array([cache_tokens], jnp.int32),
        page_table,
        jnp.array([0, cache_tokens], jnp.int32),
    )
    specs = dict(nope_spec=nope_spec, rope_spec=rope_spec)
    nope, rope = kv_cache_utils.update_sparse_mla_kv_cache(*args, **specs)
    golden = kv_cache_utils.update_sparse_mla_kv_cache_jax(*args, **specs)

    topk_idx = np.stack(
        [
            rng.choice(cache_tokens, size=topk, replace=False)
            for _ in range(query_tokens)
        ]
    ).astype(np.int32)
    q = jnp.asarray(
        rng.normal(size=(query_tokens, num_heads, NOPE_DIM + ROPE_DIM)), jnp.bfloat16
    )
    # The cache holds block-swapped codes (see `_tq_pack_sparse_mla_latent`);
    # the references dequantize the natural-order ones.
    codes, _ = tq.quantize_mla_latent(latent, tables)
    return dict(
        tables=tables,
        latent=latent,
        k_pe=k_pe,
        codes=codes,
        rope_row=rope_row,
        nope=nope,
        rope=rope,
        golden=golden,
        page_table=page_table,
        topk=jnp.asarray(topk_idx),
        q=q,
        cu_q_lens=jnp.array([0, query_tokens], jnp.int32),
        distribution=jnp.array([0, 1, 1], jnp.int32),
    )


def _reference(p, latent_rows, sm_scale, k_scale, kv_len=None):
    """f32 attention over `latent_rows[topk]` (+ fp8 k_pe), nope output only."""
    k = jnp.concatenate(
        [latent_rows.astype(jnp.float32), p["k_pe"].astype(jnp.float32) * k_scale], -1
    )[p["topk"]]  # [T, K, 576]
    s = jnp.einsum("thd,tkd->thk", p["q"].astype(jnp.float32), k) * sm_scale
    if kv_len is not None:
        s = jnp.where(jnp.arange(s.shape[-1]) < kv_len, s, -jnp.inf)
    return jnp.einsum("thk,tkd->thd", jax.nn.softmax(s, -1), k)[..., :NOPE_DIM]


def _run_tq(p, sm_scale, k_scale, topk=None, **kwargs):
    hadamard = jnp.asarray(p["tables"].hadamard)
    q_nope = tq.rotate(p["q"][..., :NOPE_DIM], hadamard).astype(jnp.bfloat16)
    q = jnp.concatenate([q_nope, p["q"][..., NOPE_DIM:]], -1)
    out = sparse_mla.sparse_ragged_paged_attention(
        q,
        p["nope"],
        p["rope"],
        p["topk"] if topk is None else topk,
        p["page_table"],
        p["cu_q_lens"],
        p["distribution"],
        sm_scale=sm_scale,
        k_scale=k_scale,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        tq_centroids=tuple(float(c) for c in p["tables"].centroids),
        **kwargs,
    )
    return tq.rotate(out[..., :NOPE_DIM], hadamard)


class SparseMlaTurboQuantTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        if jax.devices()[0].platform != "tpu":
            self.skipTest("SparseCore gather requires a TPU")

    def test_scatter_matches_xla_golden(self):
        p = _build(np.random.default_rng(1), query_tokens=16, topk=256)
        np.testing.assert_array_equal(np.asarray(p["nope"]), np.asarray(p["golden"][0]))
        np.testing.assert_array_equal(np.asarray(p["rope"]), np.asarray(p["golden"][1]))

    @parameterized.named_parameters(
        dict(testcase_name="prefill", query_tokens=128, topk=256, k_scale=1.0, kw={}),
        dict(
            testcase_name="prefill_lloyd",
            query_tokens=128,
            topk=256,
            k_scale=1.0,
            kw={},
            codebook="lloyd",
        ),
        dict(testcase_name="kscale", query_tokens=128, topk=256, k_scale=0.75, kw={}),
        dict(testcase_name="decode", query_tokens=16, topk=2048, k_scale=1.0, kw={}),
        dict(
            testcase_name="capped",
            query_tokens=64,
            topk=2048,
            k_scale=1.0,
            kw=dict(topk_cap=512),
        ),
    )
    def test_matches_dequantized_reference(
        self, query_tokens, topk, k_scale, kw, codebook="cubic"
    ):
        p = _build(
            np.random.default_rng(2),
            query_tokens=query_tokens,
            topk=topk,
            k_scale=k_scale,
            codebook=codebook,
        )
        sm_scale = (NOPE_DIM + ROPE_DIM) ** -0.5
        out = jax.block_until_ready(
            _run_tq(
                p,
                sm_scale,
                k_scale,
                gather_and_attention_chunk_size=min(64, query_tokens),
                attention_kernel_batch_size=16,
                **kw,
            )
        )
        dequant = tq.dequant_mla_latent_ref(
            p["codes"], p["rope_row"][:, tq.MLA_TQ_NORM_BYTE_OFFSET :], p["tables"]
        )
        ref_q = _reference(p, dequant, sm_scale, k_scale)
        ref_bf16 = _reference(p, p["latent"], sm_scale, k_scale)
        cos_q = _cos(out, ref_q)
        cos_bf16 = _cos(out, ref_bf16)
        print(
            f"cos vs dequant ref {cos_q.mean():.6f}/{cos_q.min():.6f}, "
            f"vs bf16 ref {cos_bf16.mean():.6f}/{cos_bf16.min():.6f}"
        )
        self.assertGreater(cos_q.min(), 0.995)
        self.assertGreater(cos_bf16.mean(), 0.98)

    def test_padded_topk(self):
        p = _build(np.random.default_rng(3), query_tokens=16, topk=256)
        sm_scale = (NOPE_DIM + ROPE_DIM) ** -0.5
        topk = p["topk"].at[:, 200:].set(-1)
        out = _run_tq(
            p,
            sm_scale,
            1.0,
            topk=topk,
            gather_and_attention_chunk_size=16,
            attention_kernel_batch_size=16,
        )
        dequant = tq.dequant_mla_latent_ref(
            p["codes"], p["rope_row"][:, tq.MLA_TQ_NORM_BYTE_OFFSET :], p["tables"]
        )
        ref = _reference(p, dequant, sm_scale, 1.0, kv_len=200)
        self.assertFalse(bool(jnp.isnan(out).any()))
        self.assertGreater(_cos(out, ref).min(), 0.995)


if __name__ == "__main__":
    absltest.main()
