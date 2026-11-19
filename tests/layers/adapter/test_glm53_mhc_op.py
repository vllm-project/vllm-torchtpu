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
"""Tests for the GLM 5.3 mHC torch ops (`custom_ops/glm53_mhc_tpu.py`)."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.custom_ops import glm53_mhc_tpu


def _to_jax(t: torch.Tensor) -> jax.Array:
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.view(torch.uint16).numpy()).view(jnp.bfloat16)
    return jnp.asarray(t.numpy())


def _to_torch(a: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a.astype(jnp.float32)))


def _make_site(hidden: int, *, seed: int = 0, **kwargs):
    torch.manual_seed(seed)
    mod = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(hidden_size=hidden, **kwargs)
    with torch.no_grad():
        mod.fn.normal_(0, 0.02)
        mod.base.normal_(0, 0.1)
        mod.scale.copy_(torch.tensor([0.5, 1.0, 2.0]))
        mod.sublayer_norm_weight.uniform_(0.5, 1.5)
    return mod


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("", "pallas"),
        ("Pallas", "pallas"),
        ("pallas", "pallas"),
        ("compiled", "compiled"),
    ],
)
def test_mhc_impl_env(monkeypatch, raw, want):
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_MHC_IMPL", raw, raising=False)
    assert glm53_mhc_tpu.glm53_mhc_impl() == want


def test_mhc_impl_env_rejects_unknown(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_GLM53_MHC_IMPL", "bogus", raising=False)
    with pytest.raises(ValueError, match="VLLM_TPU_GLM53_MHC_IMPL"):
        glm53_mhc_tpu.glm53_mhc_impl()


def test_split_fn3_torch_matches_jax_reduce_precision_bitwise():
    from vllm_torchtpu.kernels.glm53_mhc import split_fn3

    for seed, scale in ((0, 1e-4), (1, 2e-2), (2, 1.0), (3, 1e2)):
        torch.manual_seed(seed)
        fn_t = torch.randn(24, 1024, dtype=torch.float32) * scale
        got = glm53_mhc_tpu.split_fn3_torch(fn_t)
        want = split_fn3(jnp.asarray(fn_t.numpy()))
        for g, w in zip(got, want, strict=True):
            assert g.dtype == torch.bfloat16
            np.testing.assert_array_equal(
                g.view(torch.uint16).numpy(), np.asarray(w).view(np.uint16)
            )


def test_fn_split_cache_refreshes_after_in_place_update():
    hc = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(hidden_size=128, hc_mult=4)
    torch.nn.init.normal_(hc.fn, std=0.02)
    split1 = hc.prepare_pallas_weights()
    assert hc.fn_split is split1
    # An in-place update under no_grad bumps `_version`, which invalidates the
    # cached split.
    with torch.no_grad():
        hc.fn.mul_(2.0)
    split2 = hc.fn_split
    assert split2 is not split1
    for a, b in zip(split2, glm53_mhc_tpu.split_fn3_torch(hc.fn), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_compiled_path_sinkhorn_is_doubly_stochastic():
    mod = _make_site(64, hc_mult=4, hc_sinkhorn_iters=20)
    with torch.no_grad():
        mod.scale.fill_(1.0)
    streams = torch.randn(1, 4, 4, 64, dtype=torch.bfloat16)
    post, comb, normed = mod.collapse_and_norm(streams)
    assert post.shape == (1, 4, 4)
    assert comb.shape == (1, 4, 4, 4)
    assert normed.shape == (1, 4, 64)
    ones = torch.ones_like(comb.sum(dim=-1))
    torch.testing.assert_close(comb.sum(dim=-1), ones, rtol=0, atol=1e-4)
    torch.testing.assert_close(comb.sum(dim=-2), ones, rtol=0, atol=1e-4)
    assert mod.combine(streams, normed, post, comb).shape == (1, 4, 4, 64)


def test_compiled_path_matches_vllm_mhc_reference():
    """The `compiled` path == vLLM's `mhc_pre_torch` / `mhc_post_torch`.

    Chains two sites (pre -> post -> pre) like consecutive sublayers.
    """
    from vllm.model_executor.kernels.mhc import mhc_post_torch, mhc_pre_torch

    hidden, hc, iters, eps, rms_eps = 256, 4, 20, 1e-6, 1e-5
    sites = [
        _make_site(
            hidden,
            seed=i,
            hc_mult=hc,
            hc_sinkhorn_iters=iters,
            hc_eps=eps,
            rms_norm_eps=rms_eps,
        )
        for i in range(2)
    ]
    torch.manual_seed(0)
    streams = torch.randn(8, hc, hidden, dtype=torch.bfloat16)
    sub_out = torch.randn(8, hidden, dtype=torch.bfloat16)

    res_ref, res_vllm = streams, streams
    for i, mod in enumerate(sites):
        post_r, comb_r, normed_r = mod.collapse_and_norm(res_ref)
        post_v, comb_v, collapsed = mhc_pre_torch(
            res_vllm,
            mod.fn,
            mod.scale,
            mod.base,
            rms_eps,
            eps,
            eps,
            glm53_mhc_tpu.GLM53_HC_POST_MULT,
            iters,
        )
        normed_v = mod.sublayer_norm(collapsed)
        # After a seam the two residuals can differ by one bf16 ulp in a few
        # elements, which feeds the mix GEMM.
        rtol, atol = (1e-4, 1e-5) if i == 0 else (5e-3, 1e-4)
        torch.testing.assert_close(post_v.squeeze(-1), post_r, rtol=rtol, atol=atol)
        torch.testing.assert_close(comb_v, comb_r, rtol=rtol, atol=atol)
        # vLLM's pre rounds the collapse to bf16 before the norm.
        torch.testing.assert_close(
            normed_v.float(), normed_r.float(), rtol=2e-2, atol=2e-2
        )
        res_ref = mod.combine(res_ref, sub_out, post_r, comb_r)
        res_vllm = mhc_post_torch(sub_out, res_vllm, post_v, comb_v)
        torch.testing.assert_close(
            res_vllm.float(), res_ref.float(), rtol=2e-2, atol=2e-2
        )


def test_rms_norm_weighted_matches_vllm_rms_norm():
    from vllm.ir.ops.layernorm import rms_norm as vllm_rms_norm

    torch.manual_seed(0)
    x = (torch.randn(7, 512) * 3).to(torch.bfloat16)
    for w in (torch.rand(512) + 0.5, (torch.rand(512) + 0.5).bfloat16()):
        want = vllm_rms_norm(x, w, 1e-5)
        got = glm53_mhc_tpu.rms_norm_weighted(x, w, 1e-5)
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        jx = glm53_mhc_tpu._rms_norm_weighted_jax(_to_jax(x), _to_jax(w), eps=1e-5)
        torch.testing.assert_close(_to_torch(jx), want.float(), rtol=1e-2, atol=1e-2)


def test_jax_op_bodies_are_fully_annotated():
    """torch_tpu's `jax_op` rejects JAX functions with unannotated args."""
    for fn in (
        glm53_mhc_tpu._glm53_mhc_pre_norm_jax,
        glm53_mhc_tpu._glm53_mhc_post_pre_norm_jax,
        glm53_mhc_tpu._rms_norm_weighted_jax,
        glm53_mhc_tpu._mhc_mix_logits_jax,
    ):
        for name, p in inspect.signature(fn).parameters.items():
            assert p.annotation is not inspect.Parameter.empty, (fn.__name__, name)


def test_pallas_op_fakes_derive_shapes():
    res = torch.empty((8, 16384), dtype=torch.bfloat16)
    x = torch.empty((8, 4096), dtype=torch.bfloat16)
    post = torch.empty((8, 4, 1), dtype=torch.float32)
    comb = torch.empty((8, 4, 4), dtype=torch.float32)
    fn_c = torch.empty((24, 16384), dtype=torch.bfloat16)
    sc = torch.empty((3,), dtype=torch.float32)
    hb = torch.empty((24,), dtype=torch.float32)
    nw = torch.empty((4096,), dtype=torch.float32)

    pre_out = glm53_mhc_tpu._fake_mhc_pre_norm(res, fn_c, fn_c, fn_c, sc, hb, nw)
    assert [tuple(t.shape) for t in pre_out] == [(8, 4, 1), (8, 4, 4), (8, 4096)]
    assert pre_out[0].dtype == pre_out[1].dtype == torch.float32
    assert pre_out[2].dtype == torch.bfloat16

    fused_out = glm53_mhc_tpu._fake_mhc_post_pre_norm(
        x, res, post, comb, fn_c, fn_c, fn_c, sc, hb, nw
    )
    assert [tuple(t.shape) for t in fused_out] == [
        (8, 16384),
        (8, 4, 1),
        (8, 4, 4),
        (8, 4096),
    ]


def test_mhc_ops_are_built_once_per_site(monkeypatch):
    built = []

    def fake_get(rms_eps, hc_eps, iters):
        built.append((rms_eps, hc_eps, iters))
        return glm53_mhc_tpu.Glm53MhcOps(pre="pre", fused="fused", post="post")

    monkeypatch.setattr(glm53_mhc_tpu, "get_glm53_mhc_ops", fake_get)
    hc = glm53_mhc_tpu.FusedGlm53HyperConnectionTpu(
        hidden_size=64, hc_mult=4, hc_sinkhorn_iters=7, hc_eps=1e-6
    )
    assert hc.mhc_ops.pre == "pre"
    assert hc.mhc_ops.fused == "fused"
    assert built == [(1e-5, 1e-6, 7)]


@pytest.mark.tpu_test
def test_pallas_op_bodies_match_vllm_reference():
    """The `pallas` path's op bodies vs vLLM's torch reference (TPU only)."""
    if jax.default_backend() != "tpu":
        pytest.skip("Pallas mHC kernels need a TPU backend.")
    from vllm.model_executor.kernels.mhc import mhc_post_torch, mhc_pre_torch

    torch.manual_seed(0)
    hidden, hc, iters, eps, rms_eps, tokens = 4096, 4, 20, 1e-6, 1e-5, 40
    mix = (2 + hc) * hc
    fn = torch.randn(mix, hc * hidden) * 0.02
    base = torch.randn(mix) * 0.1
    scale = torch.tensor([0.5, 1.0, 2.0])
    w = torch.rand(hidden) + 0.5
    streams = torch.randn(tokens, hc, hidden).to(torch.bfloat16)
    sub_out = torch.randn(tokens, hidden).to(torch.bfloat16)
    consts = dict(rms_eps=rms_eps, hc_eps=eps, iters=iters)
    post_mult = glm53_mhc_tpu.GLM53_HC_POST_MULT

    post_r, comb_r, lay_r = mhc_pre_torch(
        streams, fn, scale, base, rms_eps, eps, eps, post_mult, iters
    )
    normed_r = glm53_mhc_tpu._rms_norm_weighted_fn(lay_r, w, rms_eps)
    res_r = mhc_post_torch(sub_out, streams, post_r, comb_r)
    post2_r, comb2_r, lay2_r = mhc_pre_torch(
        res_r, fn, scale, base, rms_eps, eps, eps, post_mult, iters
    )
    normed2_r = glm53_mhc_tpu._rms_norm_weighted_fn(lay2_r, w, rms_eps)

    flat = _to_jax(streams).reshape(tokens, hc * hidden)
    fn_split = [_to_jax(c) for c in glm53_mhc_tpu.split_fn3_torch(fn)]
    gates = (_to_jax(scale), _to_jax(base), _to_jax(w))
    post_k, comb_k, normed_k = glm53_mhc_tpu._glm53_mhc_pre_norm_jax(
        flat, *fn_split, *gates, **consts
    )
    torch.testing.assert_close(_to_torch(post_k), post_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(_to_torch(comb_k), comb_r, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        _to_torch(normed_k), normed_r.float(), rtol=2e-2, atol=2e-2
    )
    res_k, post2_k, comb2_k, normed2_k = glm53_mhc_tpu._glm53_mhc_post_pre_norm_jax(
        _to_jax(sub_out), flat, post_k, comb_k, *fn_split, *gates, **consts
    )
    torch.testing.assert_close(
        _to_torch(res_k).view(tokens, hc, hidden), res_r.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(_to_torch(post2_k), post2_r, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(_to_torch(comb2_k), comb2_r, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(
        _to_torch(normed2_k), normed2_r.float(), rtol=2e-2, atol=2e-2
    )
