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
from jax.sharding import Mesh

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
SM_SCALE = (NOPE_DIM + ROPE_DIM) ** -0.5


def _skip_if_not_tpu():
    if jax.devices()[0].platform != "tpu":
        pytest.skip("SparseCore gather requires a TPU")


def _cos(a, b):
    a = np.asarray(a, np.float32).reshape(-1, a.shape[-1])
    b = np.asarray(b, np.float32).reshape(-1, b.shape[-1])
    return np.sum(a * b, -1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def _build(rng, *, query_tokens, topk, cache_tokens=8192, k_scale=1.0):
    pages = cache_tokens // PAGE_SIZE
    tables = tq.make_mla_latent_tables()
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
        KVCacheType.ROPE, KVCacheLayout.SPARSECORE, pages, PAGE_SIZE, ROPE_DIM + 2
    )
    page_table = jnp.asarray(rng.permutation(pages).astype(np.int32))
    codes_swapped, rope_row = attention_interface._tq_pack_sparse_mla_latent(
        latent, k_pe, tables
    )
    args = (
        jnp.zeros(nope_spec.shape, nope_spec.jax_dtype),
        jnp.zeros(rope_spec.shape, rope_spec.jax_dtype),
        codes_swapped,
        rope_row,
        jnp.array([cache_tokens], jnp.int32),
        page_table,
        jnp.array([0, cache_tokens], jnp.int32),
    )
    nope, rope = kv_cache_utils.update_sparse_mla_kv_cache(
        *args, nope_spec=nope_spec, rope_spec=rope_spec
    )
    topk_idx = np.stack(
        [
            rng.choice(cache_tokens, size=topk, replace=False)
            for _ in range(query_tokens)
        ]
    ).astype(np.int32)
    q = jnp.asarray(
        rng.normal(size=(query_tokens, 8, NOPE_DIM + ROPE_DIM)), jnp.bfloat16
    )
    return dict(
        tables=tables,
        latent=latent,
        k_pe=k_pe,
        rope_row=rope_row,
        nope_spec=nope_spec,
        rope_spec=rope_spec,
        nope=nope,
        rope=rope,
        scatter_args=args,
        page_table=page_table,
        topk=jnp.asarray(topk_idx),
        q=q,
        cu_q_lens=jnp.array([0, query_tokens], jnp.int32),
        distribution=jnp.array([0, 1, 1], jnp.int32),
    )


def _reference(p, latent_rows, k_scale, kv_len=None):
    k = jnp.concatenate(
        [latent_rows.astype(jnp.float32), p["k_pe"].astype(jnp.float32) * k_scale], -1
    )[p["topk"]]
    s = jnp.einsum("thd,tkd->thk", p["q"].astype(jnp.float32), k) * SM_SCALE
    if kv_len is not None:
        s = jnp.where(jnp.arange(s.shape[-1]) < kv_len, s, -jnp.inf)
    return jnp.einsum("thk,tkd->thd", jax.nn.softmax(s, -1), k)[..., :NOPE_DIM]


def test_scatter_matches_xla_golden():
    _skip_if_not_tpu()
    p = _build(np.random.default_rng(1), query_tokens=16, topk=256, cache_tokens=1024)
    golden = kv_cache_utils.update_sparse_mla_kv_cache_jax(
        *p["scatter_args"], nope_spec=p["nope_spec"], rope_spec=p["rope_spec"]
    )
    np.testing.assert_array_equal(np.asarray(p["nope"]), np.asarray(golden[0]))
    np.testing.assert_array_equal(np.asarray(p["rope"]), np.asarray(golden[1]))


@pytest.mark.parametrize(
    ("query_tokens", "topk", "k_scale", "valid_k", "kw"),
    [
        pytest.param(64, 256, 0.75, 200, {}, id="non_split_padded_kscale"),
        pytest.param(16, 2048, 1.0, None, dict(split_q=True), id="decode_split_q"),
        pytest.param(
            64, 2048, 1.0, None, dict(split_q=True, topk_cap=512), id="capped_split_q"
        ),
    ],
)
def test_matches_dequantized_reference(query_tokens, topk, k_scale, valid_k, kw):
    _skip_if_not_tpu()
    p = _build(
        np.random.default_rng(2), query_tokens=query_tokens, topk=topk, k_scale=k_scale
    )
    hadamard = jnp.asarray(p["tables"].hadamard)
    q_nope = tq.rotate(p["q"][..., :NOPE_DIM], hadamard).astype(jnp.bfloat16)
    split_q = kw.get("split_q", False)
    call_kw = {k: v for k, v in kw.items() if k != "split_q"}
    if split_q:
        q, call_kw["q_pe"] = q_nope, p["q"][..., NOPE_DIM:]
    else:
        q = jnp.concatenate([q_nope, p["q"][..., NOPE_DIM:]], -1)
    topk_in = p["topk"].at[:, valid_k:].set(-1) if valid_k else p["topk"]
    out = jax.block_until_ready(
        sparse_mla.sparse_ragged_paged_attention(
            q,
            p["nope"],
            p["rope"],
            topk_in,
            p["page_table"],
            p["cu_q_lens"],
            p["distribution"],
            sm_scale=SM_SCALE,
            k_scale=k_scale,
            cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
            turboquant=True,
            gather_and_attention_chunk_size=min(64, query_tokens),
            attention_kernel_batch_size=16,
            **call_kw,
        )
    )
    assert not bool(jnp.isnan(out).any())
    if split_q:
        assert out.shape[-1] == NOPE_DIM
    out = tq.rotate(out[..., :NOPE_DIM], hadamard)

    codes, _ = tq.quantize_mla_latent(p["latent"], p["tables"])
    dequant = tq.dequant_mla_latent_ref(
        codes, p["rope_row"][:, tq.MLA_TQ_NORM_BYTE_OFFSET :], p["tables"]
    )
    assert _cos(out, _reference(p, dequant, k_scale, kv_len=valid_k)).min() > 0.995
    assert _cos(out, _reference(p, p["latent"], k_scale, kv_len=valid_k)).mean() > 0.98


def test_sparse_mla_attention_interface_prerotated():
    _skip_if_not_tpu()
    p = _build(
        np.random.default_rng(4),
        query_tokens=64,
        topk=256,
        cache_tokens=512,
        k_scale=0.8,
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))
    ql_nope, q_pe = p["q"][..., :NOPE_DIM], p["q"][..., NOPE_DIM:]
    hadamard = jnp.asarray(p["tables"].hadamard)

    def run(q_in, prerot: bool):
        return attention_interface.sparse_mla_attention(
            q_in,
            q_pe,
            p["latent"][:64],
            p["k_pe"][:64],
            p["nope"],
            p["rope"],
            p["topk"],
            jnp.array([512], jnp.int32),
            p["page_table"],
            jnp.array([0, 64], jnp.int32),
            jnp.array([0, 1, 1], jnp.int32),
            mesh,
            p["nope_spec"],
            p["rope_spec"],
            sm_scale=SM_SCALE,
            k_scale=0.8,
            tq_tables=p["tables"],
            tq_prerotated=prerot,
        )[2]

    out_unrot = run(ql_nope, False)
    out_rot = run(tq.rotate(ql_nope, hadamard).astype(ql_nope.dtype), True)
    assert _cos(out_unrot, tq.rotate(out_rot, hadamard)).min() > 0.999
