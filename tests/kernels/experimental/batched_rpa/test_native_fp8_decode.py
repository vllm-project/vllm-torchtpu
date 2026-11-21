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
no request owns, is an fp8 NaN, so the outputs also show that those bytes stay
out of the native matmuls (V is masked past each request's KV length on load).
test_decode_fp8_kv also runs the row-scaled modes on an e5m2 KV cache.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import test_util as jtu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, wrapper

pytestmark = pytest.mark.tpu_test

_NQ, _NKV, _HD, _PAGE, _QLEN = 4, 1, 256, 128, 4
_SM_SCALE = 1.0 / 16.0
_FP8 = jnp.float8_e4m3fn
_E5M2 = jnp.float8_e5m2


def _cdiv(a, b):
    return (a + b - 1) // b


def _rand_fp8(key, shape, dtype=_FP8):
    """Random `dtype` values: random sign and mantissa bits, and one of the
    four exponents 2^-2 to 2^1."""
    nmant, bias = jnp.finfo(dtype).nmant, 1 - jnp.finfo(dtype).minexp
    bits = jax.random.bits(key, shape, jnp.uint8)
    sign_mant = bits & (0x80 | ((1 << nmant) - 1))
    byte = sign_mant | ((((bits >> nmant) & 0b11) + bias - 2) << nmant)
    return jax.lax.bitcast_convert_type(byte.astype(jnp.uint8), dtype)


def _inputs(kv_lens_list, seed=0, dtype=_FP8):
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
    cache = jax.lax.bitcast_convert_type(_rand_fp8(k1, shape, dtype), jnp.uint8)
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
    k = _rand_fp8(k3, (t, _NKV, _HD), dtype)
    v = _rand_fp8(k4, (t, _NKV, _HD), dtype)
    kv_lens = np.zeros((max_num_seqs,), np.int32)
    kv_lens[:reqs] = kv_lens_list
    cu = np.full((max_num_seqs + 1,), reqs * _QLEN, np.int32)
    cu[: reqs + 1] = np.arange(reqs + 1) * _QLEN
    return dict(
        q=q,
        k=k,
        v=v,
        kv_cache=jax.lax.bitcast_convert_type(cache, dtype),
        kv_lens=jnp.asarray(kv_lens),
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=jnp.asarray(cu),
        distribution=jnp.array([reqs, reqs, reqs], jnp.int32),
        pps=pps,
    )


def _reference(inp, kv_lens_list, k_scale=1.0, v_scale=1.0):
    pps = inp["pps"]
    outs = []
    for i, kv_len in enumerate(kv_lens_list):
        pages = inp["page_indices"][i * pps : (i + 1) * pps]
        g = inp["kv_cache"][pages].astype(jnp.float32)
        kk = g[:, 0].transpose(0, 3, 1, 2).reshape(pps * _PAGE, _HD)
        vv = g[:, 1].transpose(0, 3, 1, 2).reshape(pps * _PAGE, _HD)
        sl = slice(i * _QLEN, (i + 1) * _QLEN)
        pos = kv_len - _QLEN + jnp.arange(_QLEN)
        kk = kk.at[pos].set(inp["k"][sl, 0].astype(jnp.float32))[:kv_len] * k_scale
        vv = vv.at[pos].set(inp["v"][sl, 0].astype(jnp.float32))[:kv_len] * v_scale
        s = jnp.einsum("qhd,kd->hqk", inp["q"][sl].astype(jnp.float32), kk)
        mask = jnp.arange(kv_len)[None, None, :] <= pos[None, :, None]
        p = jax.nn.softmax(jnp.where(mask, s * _SM_SCALE, -jnp.inf), axis=-1)
        outs.append(jnp.einsum("hqk,kd->qhd", p, vv))
    return np.asarray(jnp.concatenate(outs, axis=0))


_ROW, _SPLIT = configs.Fp8Operand.ROW, configs.Fp8Operand.SPLIT
# tpu7x quick bench (ctx 33k): rel L2 ~0.02 with every operand off and with a
# row-scaled PV, ~2x that with both operands split, ~0.08 with both
# row-scaled (q at 3 mantissa bits).
_MODES = {
    "off": (configs.NativeFp8(), 0.03),
    "pv": (configs.NativeFp8(pv=_ROW), 0.03),
    "qk_split_pv_split": (configs.NativeFp8(qk=_SPLIT, pv=_SPLIT), 0.06),
    "qk_pv": (configs.NativeFp8(qk=_ROW, pv=_ROW), 0.12),
}
# test_decode_fp8_kv's (KV dtype, mode, bound). e5m2 keeps 2 mantissa bits to
# e4m3's 3, so each operand a native mode casts to it is rounded with twice the
# step; its bounds are the e4m3 ones doubled. SPLIT is refused on e5m2.
_KV_CASES = [
    pytest.param(_FP8, mode, tol, id=f"e4m3-{mode}")
    for mode, (_, tol) in _MODES.items()
] + [
    pytest.param(_E5M2, mode, 2 * _MODES[mode][1], id=f"e5m2-{mode}")
    for mode in ("pv", "qk_pv")
]


@pytest.mark.parametrize("mode", ["pv", "qk_split_pv_split", "qk_pv"])
def test_decode_fp8_kv_with_kv_scales(mode):
    """k_scale and v_scale other than 1 reach the native QK scores and the
    native PV output: the output matches an f32 reference that multiplies
    K by k_scale and V by v_scale."""
    native_fp8, tol = _MODES[mode]
    if not jtu.is_device_tpu_at_least(version=7):
        pytest.skip("Expect TPU v7+ (native fp8 MXU)")
    kv_lens_list = [5000, 9000 + 37, 131]
    inp = _inputs(kv_lens_list)
    ref = _reference(inp, kv_lens_list, k_scale=0.5, v_scale=0.75)
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
        k_scale=0.5,
        v_scale=0.75,
        kv_layout=configs.KVLayout.SEQ_ALONG_LANE,
        decode_query_size=_QLEN,
        native_fp8=native_fp8,
        tuned_tiles=True,
        skip_empty_mixed=True,
    )
    out = np.asarray(out[: len(kv_lens_list) * _QLEN].astype(jnp.float32))
    assert not np.isnan(out).any()
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel < tol, rel


@pytest.mark.parametrize(("kv_dtype", "mode", "tol"), _KV_CASES)
@pytest.mark.parametrize(
    "kv_lens_list",
    [[300], [5000, 9000 + 37, 131], [33000 + 61] + [4096 * 3] * 4],
)
def test_decode_fp8_kv(kv_dtype, mode, tol, kv_lens_list):
    native_fp8 = _MODES[mode][0]
    if not jtu.is_device_tpu_at_least(version=7):
        pytest.skip("Expect TPU v7+ (native fp8 MXU)")
    inp = _inputs(kv_lens_list, dtype=kv_dtype)
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
        # The tiles and call structure of the Qwen3.5 decode shapes these modes
        # are tuned for.
        tuned_tiles=True,
        skip_empty_mixed=True,
    )
    out = np.asarray(out[: len(kv_lens_list) * _QLEN].astype(jnp.float32))
    assert not np.isnan(out).any()
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    assert rel < tol, rel
