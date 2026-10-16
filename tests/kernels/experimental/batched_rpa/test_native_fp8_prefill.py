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
"""Prefill (RPAm) with fp8 KV: bf16 matmuls vs native fp8 QK / PV.

Qwen3.5 TP8 per-device geometry (4 q heads, 1 KV head, head_dim 256), HND
(SEQ_ALONG_LANE) fp8 e4m3 KV, chunks of several hundred tokens on top of
cached KV, lengths that end mid-block and mid-page. Every byte past a
request's KV length, and every page no request owns, is an fp8 NaN, so the
static PV scale is also checked against the unattended V lanes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

_NQ, _NKV, _HD, _PAGE = 4, 1, 256, 128
_SM_SCALE = 1.0 / 16.0
_FP8 = jnp.float8_e4m3fn


def _cdiv(a, b):
    return (a + b - 1) // b


def _rand_fp8(key, shape):
    bits = jax.random.bits(key, shape, jnp.uint8)
    byte = (bits & 0b10000111) | ((((bits >> 3) & 0b11) + 5) << 3)
    return jax.lax.bitcast_convert_type(byte.astype(jnp.uint8), _FP8)


def _inputs(reqs, seed=0):
    """reqs: (cached tokens, new tokens) per request."""
    k1, k2, k3, k4 = jax.random.split(jax.random.key(seed), 4)
    kv_lens_list = [c + n for c, n in reqs]
    q_lens = [n for _, n in reqs]
    max_num_seqs = max(8, _cdiv(len(reqs), 8) * 8)
    pps = _cdiv(max(kv_lens_list), _PAGE)
    used = len(reqs) * pps
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
    for i, (cached, _) in enumerate(reqs):
        # Past the cached prefix the kernel writes the new K/V; leave NaN
        # in the tail of the last cached page and in every later page.
        first = cached // _PAGE
        for j in range(first, pps):
            p = int(page_indices[i * pps + j])
            lo = cached % _PAGE if j == first else 0
            cache = cache.at[p, :, :, :, lo:].set(nan)
    t = sum(q_lens)
    q = (jax.random.normal(k2, (t, _NQ, _HD)) * 2.0).astype(jnp.bfloat16)
    k = _rand_fp8(k3, (t, _NKV, _HD))
    v = _rand_fp8(k4, (t, _NKV, _HD))
    kv_lens = np.zeros((max_num_seqs,), np.int32)
    kv_lens[: len(reqs)] = kv_lens_list
    cu = np.full((max_num_seqs + 1,), t, np.int32)
    cu[: len(reqs) + 1] = np.concatenate([[0], np.cumsum(q_lens)])
    n = len(reqs)
    return dict(
        q=q,
        k=k,
        v=v,
        kv_cache=jax.lax.bitcast_convert_type(cache, _FP8),
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu),
        distribution=jnp.array([0, 0, n], jnp.int32),
        pps=pps,
        cu=cu,
    )


def _reference(inp, reqs):
    pps = inp["pps"]
    outs = []
    for i, (cached, new) in enumerate(reqs):
        kv_len = cached + new
        pages = inp["page_indices"][i * pps : (i + 1) * pps]
        g = inp["kv_cache"][pages].astype(jnp.float32)
        kk = g[:, 0].transpose(0, 3, 1, 2).reshape(pps * _PAGE, _HD)
        vv = g[:, 1].transpose(0, 3, 1, 2).reshape(pps * _PAGE, _HD)
        sl = slice(int(inp["cu"][i]), int(inp["cu"][i + 1]))
        pos = cached + jnp.arange(new)
        kk = kk.at[pos].set(inp["k"][sl, 0].astype(jnp.float32))[:kv_len]
        vv = vv.at[pos].set(inp["v"][sl, 0].astype(jnp.float32))[:kv_len]
        s = jnp.einsum("qhd,kd->hqk", inp["q"][sl].astype(jnp.float32), kk)
        mask = jnp.arange(kv_len)[None, None, :] <= pos[None, :, None]
        p = jax.nn.softmax(jnp.where(mask, s * _SM_SCALE, -jnp.inf), axis=-1)
        outs.append(jnp.einsum("hqk,kd->qhd", p, vv))
    return np.asarray(jnp.concatenate(outs, axis=0))


@pytest.mark.parametrize("native_fp8_prefill", ["", "pv", "pvs", "qks", "qkspvs"])
@pytest.mark.parametrize(
    "reqs",
    [[(4096 + 37, 300)], [(0, 1024), (9000 + 61, 513), (1280, 7)]],
)
def test_prefill_fp8_kv(native_fp8_prefill, reqs):
    if not jtu.is_device_tpu_at_least(version=7):
        pytest.skip("Expect TPU v7+ (native fp8 MXU)")
    inp = _inputs(reqs)
    ref = _reference(inp, reqs)
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
        native_fp8_prefill=native_fp8_prefill,
    )
    t = sum(n for _, n in reqs)
    out = np.asarray(out[:t].astype(jnp.float32))
    assert not np.isnan(out).any()
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    # q at 3 mantissa bits dominates the qk rows, as in the decode test.
    tol = {"": 0.03, "pv": 0.03, "pvs": 0.03, "qks": 0.12, "qkspvs": 0.12}
    assert rel < tol[native_fp8_prefill], rel
