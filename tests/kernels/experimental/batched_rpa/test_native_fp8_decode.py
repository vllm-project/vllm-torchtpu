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
"""Decode (RPAd) with fp8 KV: default bf16 matmuls vs native fp8 matmuls.

Qwen3.5 TP8 per-device geometry (4 q heads, 1 KV head, head_dim 256), HND
(SEQ_ALONG_LANE) fp8 e4m3 KV, 4 query tokens per request, KV lengths that end
mid-block and mid-page. Every byte past a request's KV length, and every page
no request owns, is an fp8 NaN, so this also covers the per-step zeroing of
unattended V lanes that replaced the per-call zeroing of all KV VMEM.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_NQ, _NKV, _HD, _PAGE, _QLEN = 4, 1, 256, 128, 4
_SM_SCALE = 1.0 / 16.0
_FP8 = jnp.float8_e4m3fn


def _cdiv(a, b):
    return (a + b - 1) // b


def _rand_fp8(key, shape):
    bits = jax.random.bits(key, shape, jnp.uint8)
    byte = (bits & 0b10000111) | ((((bits >> 3) & 0b11) + 5) << 3)
    return jax.lax.bitcast_convert_type(byte.astype(jnp.uint8), _FP8)


def _inputs(kv_lens_list, seed=0):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(seed), 4)
    reqs = len(kv_lens_list)
    max_num_seqs = max(8, _cdiv(reqs, 8) * 8)
    pps = _cdiv(max(kv_lens_list), _PAGE)
    used = reqs * pps
    num_pages = used + 8
    perm = np.random.default_rng(seed).permutation(num_pages)[:used].astype(np.int32)
    page_indices = np.full((max_num_seqs * pps,), num_pages - 1, np.int32)
    page_indices[:used] = perm
    shape = (num_pages, 2 * _NKV, _HD // 4, 4, _PAGE)
    cache = jax.lax.bitcast_convert_type(_rand_fp8(k1, shape), jnp.uint8)
    nan = jnp.uint8(0x7F)
    owned = np.zeros(num_pages, bool)
    owned[perm] = True
    cache = jnp.where(jnp.asarray(owned)[:, None, None, None, None], cache, nan)
    for i, kv_len in enumerate(kv_lens_list):
        if kv_len % _PAGE:
            p = int(page_indices[i * pps + (kv_len - 1) // _PAGE])
            cache = cache.at[p, :, :, :, kv_len % _PAGE :].set(nan)
    t = reqs * _QLEN
    q = (jax.random.normal(k2, (t, _NQ, _HD)) * 2.0).astype(jnp.bfloat16)
    k = _rand_fp8(k3, (t, _NKV, _HD))
    v = _rand_fp8(k4, (t, _NKV, _HD))
    kv_lens = np.zeros((max_num_seqs,), np.int32)
    kv_lens[:reqs] = kv_lens_list
    cu = np.full((max_num_seqs + 1,), reqs * _QLEN, np.int32)
    cu[: reqs + 1] = np.arange(reqs + 1) * _QLEN
    return dict(
        q=q,
        k=k,
        v=v,
        kv_cache=jax.lax.bitcast_convert_type(cache, _FP8),
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu),
        distribution=jnp.array([reqs, reqs, reqs], jnp.int32),
        pps=pps,
    )


def _reference(inp, kv_lens_list):
    pps = inp["pps"]
    outs = []
    for i, kv_len in enumerate(kv_lens_list):
        pages = inp["page_indices"][i * pps : (i + 1) * pps]
        g = inp["kv_cache"][pages].astype(jnp.float32)
        kk = g[:, 0].transpose(0, 3, 1, 2).reshape(pps * _PAGE, _HD)
        vv = g[:, 1].transpose(0, 3, 1, 2).reshape(pps * _PAGE, _HD)
        sl = slice(i * _QLEN, (i + 1) * _QLEN)
        pos = kv_len - _QLEN + jnp.arange(_QLEN)
        kk = kk.at[pos].set(inp["k"][sl, 0].astype(jnp.float32))[:kv_len]
        vv = vv.at[pos].set(inp["v"][sl, 0].astype(jnp.float32))[:kv_len]
        s = jnp.einsum("qhd,kd->hqk", inp["q"][sl].astype(jnp.float32), kk)
        mask = jnp.arange(kv_len)[None, None, :] <= pos[None, :, None]
        p = jax.nn.softmax(jnp.where(mask, s * _SM_SCALE, -jnp.inf), axis=-1)
        outs.append(jnp.einsum("hqk,kd->qhd", p, vv))
    return np.asarray(jnp.concatenate(outs, axis=0))


@pytest.mark.parametrize("native_fp8", ["", "pv", "qk2pv2", "qkpv"])
@pytest.mark.parametrize(
    "kv_lens_list",
    [[300], [5000, 9000 + 37, 131], [33000 + 61] + [4096 * 3] * 4],
)
def test_decode_fp8_kv(native_fp8, kv_lens_list):
    if not jtu.is_device_tpu_at_least(version=7):
        pytest.skip("Expect TPU v7+ (native fp8 MXU)")
    inp = _inputs(kv_lens_list)
    ref = _reference(inp, kv_lens_list)
    out, _ = wrapper.ragged_paged_attention(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["kv_cache"],
        inp["kv_lens"],
        inp["page_indices"],
        inp["cu_q_lens"],
        inp["distribution"],
        sm_scale=_SM_SCALE,
        k_scale=1.0,
        v_scale=1.0,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=_QLEN,
        native_fp8=native_fp8,
    )
    out = np.asarray(out[: len(kv_lens_list) * _QLEN].astype(jnp.float32))
    assert not np.isnan(out).any()
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    # tpu7x quick bench (ctx 33k): rel L2 ~0.02 for the default and "pv",
    # ~2x that for "qk2pv2", ~0.08 for "qkpv" (q at 3 mantissa bits).
    tol = {"": 0.03, "pv": 0.03, "qk2pv2": 0.06, "qkpv": 0.12}[native_fp8]
    assert rel < tol, rel
