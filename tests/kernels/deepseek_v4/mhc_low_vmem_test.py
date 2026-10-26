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
"""VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS and VLLM_TPU_DSV4_MHC_SEAM_FN_HBM
change only where the fused seam's buffers live, never its math: every
output must be bit-identical to the default seam, at decode sizes (where the
knobs apply) and above the cap (where they must not)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.mhc import fused_post_pre_kernel as K
from vllm_torchtpu.kernels.deepseek_v4.mhc import utils

H, M = 7168, 4
M3 = M * (M + 2)
EPS = 1e-6


def _tpu():
    if jax.default_backend() != "tpu":
        pytest.skip("needs a TPU")


def _operands(T, seed):
    rng = np.random.default_rng(seed)

    def act(shape):
        x = rng.standard_normal(shape).astype(np.float32)
        return jnp.asarray(np.where(rng.random(shape) < 1e-3, x * 30.0, x), jnp.bfloat16)

    x, res = act((T, H)), act((T, M, H))
    post = jnp.asarray(2.0 / (1.0 + np.exp(-rng.standard_normal((T, M, 1)))), jnp.float32)
    c = np.exp(rng.standard_normal((T, M, M)))
    for _ in range(20):
        c = c / c.sum(-1, keepdims=True)
        c = c / c.sum(-2, keepdims=True)
    fn = utils.cat_fn3(jnp.asarray(rng.standard_normal((M3, M * H)) * 0.02, jnp.float32))
    sc = jnp.asarray(1.0 + 0.1 * rng.standard_normal(3), jnp.float32)
    hb = jnp.asarray(0.5 * rng.standard_normal(M3), jnp.float32)
    nw = jnp.asarray(rng.uniform(0.5, 1.5, (1, H)), jnp.bfloat16)
    return x, res, post, jnp.asarray(c, jnp.float32), fn, sc, hb, nw


def _seam(ops, small_block, fn_chunks, **kw):
    x, res, post, comb, fn, sc, hb, nw = ops
    return K.mhc_fused_post_pre(
        x, res, post, comb, fn, sc, hb, EPS, EPS, EPS, 2.0, 20, token_block_size=128,
        gates_in_kernel=True, gates_max_tokens=0, norm_weight=nw, norm_eps=EPS,
        small_block=small_block, fn_chunks=fn_chunks, **kw)


@pytest.mark.parametrize("small_block,fn_chunks", [(16, 3), (8, 2)])
@pytest.mark.parametrize("T", [1, 4, 8, 16, 24, 32, 64])
def test_low_vmem_and_fn_hbm_are_bit_identical(monkeypatch, T, small_block, fn_chunks):
    _tpu()
    ops = _operands(T, seed=T)
    monkeypatch.delenv("VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS", raising=False)
    want = _seam(ops, small_block, fn_chunks, vmem_limit_bytes=utils.DEFAULT_VMEM_LIMIT_BYTES)
    monkeypatch.setenv("VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS", "32")
    for kw in ({}, {"fn_hbm": True}):
        got = _seam(ops, small_block, fn_chunks, **kw)
        for name, w, g in zip(("residual", "post_mix", "comb_mix", "layer"), want, got):
            w, g = np.ascontiguousarray(np.asarray(w)), np.ascontiguousarray(np.asarray(g))
            assert np.array_equal(w.view(np.uint8), g.view(np.uint8)), (T, kw, name, int((w != g).sum()))


def test_cap_must_be_a_small_power_of_two(monkeypatch):
    _tpu()
    ops = _operands(8, seed=0)
    monkeypatch.setenv("VLLM_TPU_DSV4_MHC_LOW_VMEM_MAX_TOKENS", "24")
    with pytest.raises(ValueError):
        _seam(ops, 16, 3)
