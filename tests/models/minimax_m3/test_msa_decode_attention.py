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
"""`msa_attention_lse` decode calls (<= DECODE_TOKENS tokens, one grid step
per token, no work list) against the tiled kernel (interpret mode)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.models.minimax_m3.packing import pack, rounded
from vllm_torchtpu.models.minimax_m3.kernels import msa_attention_lse, msa_fp4

PAGE = 128
D = 128
PAGES = 48
TOPK = 16


def _selection(rng, num_tokens, num_valid):
    """Verify-window-like picks: tokens of a request share most pages; each
    token has `num_valid` distinct local keys, the rest -1."""
    selected = np.full((num_tokens, TOPK), -1, np.int32)
    for t in range(num_tokens):
        base = (t // 4) * 7
        picks = rng.choice(np.arange(base, base + 2 * TOPK) % PAGES, num_valid, False)
        selected[t, :num_valid] = picks
        rng.shuffle(selected[t])
    local_page = selected[np.arange(num_tokens), np.argmax(selected >= 0, axis=1)]
    local_limit = rng.integers(1, PAGE + 1, num_tokens).astype(np.int32)
    return selected, local_page, local_limit


def _caches(rng, dtype, heads=1):
    if dtype == "fp4":
        nib = rng.integers(0, 16, (PAGES, PAGE, 2 * D * heads)).astype(np.uint8)
        scales = rng.random((PAGES, 2 * D * heads)) * 0.1  # folded K / V scales
        return msa_fp4.pack_rows(jnp.asarray(nib), scales=jnp.asarray(scales))
    rows = rounded(rng.standard_normal((PAGES, PAGE, 2 * D * heads)), dtype)
    return pack(rows, dtype)


def _run(fn, monkeypatch, tiled, *args, **kwargs):
    if tiled:
        monkeypatch.setattr(msa_attention_lse, "DECODE_TOKENS", 0)
    out = fn(*args, **kwargs)
    monkeypatch.setattr(msa_attention_lse, "DECODE_TOKENS", 64)
    return [np.asarray(x.astype(jnp.float32)) for x in out]


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn, "fp4"])
@pytest.mark.parametrize("num_tokens,num_valid", [(8, 8), (16, 5), (16, 0), (3, 16)])
def test_decode_matches_tiled(monkeypatch, dtype, num_tokens, num_valid):
    rng = np.random.default_rng(num_tokens * 31 + num_valid)
    selected, local_page, local_limit = _selection(rng, num_tokens, num_valid)
    kv = _caches(rng, dtype)
    q = jnp.asarray(rng.standard_normal((num_tokens, 16, D))).astype(jnp.bfloat16)
    table = jnp.asarray(rng.permutation(PAGES).astype(np.int32))
    args = (q, kv, jnp.asarray(selected), jnp.asarray(local_page), jnp.asarray(local_limit), table)
    kw = dict(sm_scale=D**-0.5, interpret=True)
    dec = _run(msa_attention_lse.msa_sparse_attention_lse.__wrapped__, monkeypatch, False, *args, **kw)
    til = _run(msa_attention_lse.msa_sparse_attention_lse.__wrapped__, monkeypatch, True, *args, **kw)
    np.testing.assert_allclose(dec[0], til[0], atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(dec[1], til[1], atol=1e-3, rtol=1e-4)
    if num_valid == 0:
        assert np.all(np.isneginf(dec[1])) and not np.any(dec[0])


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, "fp4"])
def test_decode_heads_matches_tiled(monkeypatch, dtype):
    rng = np.random.default_rng(7)
    heads, num_tokens = 4, 16
    sel = [_selection(rng, num_tokens, 6) for _ in range(heads)]
    selected = np.stack([s[0] for s in sel], axis=1)  # [T, G, K]
    local_page, local_limit = sel[0][1], sel[0][2]
    kv = _caches(rng, dtype, heads)
    q = jnp.asarray(rng.standard_normal((num_tokens, heads * 16, D))).astype(jnp.bfloat16)
    table = jnp.asarray(rng.permutation(PAGES).astype(np.int32))
    args = (q, kv, jnp.asarray(selected), jnp.asarray(local_page), jnp.asarray(local_limit), table)
    kw = dict(sm_scale=D**-0.5, interpret=True)
    fn = msa_attention_lse.msa_sparse_attention_lse_heads.__wrapped__
    dec = _run(fn, monkeypatch, False, *args, **kw)
    til = _run(fn, monkeypatch, True, *args, **kw)
    np.testing.assert_allclose(dec[0], til[0], atol=2e-2, rtol=2e-2)
    np.testing.assert_allclose(dec[1], til[1], atol=1e-3, rtol=1e-4)
