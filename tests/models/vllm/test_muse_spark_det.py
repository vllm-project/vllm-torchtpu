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
"""Batch-invariance unit tests for Muse Spark Open (VLLM_TPU_MUSE_BATCH_INVARIANT)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.batch_invariant import matmul as det_mm
from vllm_torchtpu.kernels.batch_invariant import rms_norm as det_rms
from vllm_torchtpu.kernels.batch_invariant.gmm_tiling import batch_invariant_tiling
from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    Dimensions,
    InputConfigs,
    calculate_tiling,
)
from vllm_torchtpu.models.vllm.muse_spark import batch_invariant as bi
from vllm_torchtpu.models.vllm.muse_spark import det_collectives


@pytest.mark.parametrize("kdim", [1024, 4096, 8192, 1280])
def test_tile_k_depends_on_k_only(kdim: int) -> None:
    tks = {det_mm.default_tiles(m, kdim, 4096, 2, 2, 2)[2] for m in (1, 8, 300, 4096)}
    assert tks == {det_mm.default_tile_k(kdim)}
    assert kdim % tks.pop() == 0


def test_tiles_fit_vmem_and_lane_align() -> None:
    for m in (1, 64, 4096):
        for kdim, n in ((8192, 25344), (8192, 256), (1024, 8192)):
            tm, tn, tk = det_mm.default_tiles(m, kdim, n, 4, 4, 4)
            assert tm % 8 == 0 and n % tn == 0 and tn % 128 == 0
            assert (
                2 * tm * tk * 4 + 2 * tk * tn * 4 + tm * tn * 12
                <= det_mm.vmem_budget_bytes()
            )


@pytest.mark.parametrize("rhs_nk", [False, True])
def test_det_matmul_rows_independent_of_batch(rhs_nk: bool) -> None:
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((40, 512)), jnp.float32)
    w = jnp.asarray(rng.standard_normal((512, 256)), jnp.float32)
    wl = w.T if rhs_nk else w
    ref = np.asarray(jnp.dot(x, w, precision=jax.lax.Precision.HIGHEST))
    kw = dict(rhs_nk=rhs_nk, interpret=True, tile_k=128, highest=True)
    full = np.asarray(det_mm.det_matmul(x, wl, **kw))
    np.testing.assert_allclose(full, ref, rtol=1e-5, atol=1e-4)
    # Same rows alone, at an offset, and with other tile_m/tile_n: bit equal.
    for sub, tm, tn in ((x[:3], 8, 128), (x[5:21], 16, 256), (x, 32, 128)):
        got = np.asarray(det_mm.det_matmul(sub, wl, tile_m=tm, tile_n=tn, **kw))
        start = int(np.argmax(np.all(np.asarray(x) == np.asarray(sub[:1]), axis=1)))
        np.testing.assert_array_equal(got, full[start : start + sub.shape[0]])


def test_gmm_tile_k_fixed_across_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_torchtpu.kernels.megablox import gmm_v2

    info = SimpleNamespace(mxu_column_size=256, num_lanes=128)
    monkeypatch.setattr(gmm_v2.pltpu, "get_tpu_info", lambda: info)
    lhs = InputConfigs(None, 4096, jnp.dtype(jnp.bfloat16))
    fp4 = jnp.dtype(jnp.float4_e2m1fn)
    rhs = InputConfigs(fp4, 16, fp4, has_scale=True)
    tiles = []
    for m in (8, 64, 512, 4096, 65536):
        dims = Dimensions(m, 4096, 8192, 32, 1, 8)
        t = batch_invariant_tiling(dims, lhs, rhs, 64 * 1024 * 1024, "silu")
        base = calculate_tiling(dims, lhs, rhs, 64 * 1024 * 1024, "silu")
        assert t.tile_n == base.tile_n
        if m <= 1024:
            assert dataclasses.replace(t, tile_k=base.tile_k) == base
        else:
            assert t.bucket_base == 128 and t.tile_m % t.bucket_base == 0
            assert t.tile_m == (128 if m <= 16384 else (base.tile_m // 128) * 128)
        tiles.append(t.tile_k)
    assert len(set(tiles)) == 1

    # Verify VMEM-constrained case where calculate_tiling lands on tile_m=192:
    # batch_invariant_tiling must round tile_m down to 128 so tile_m % bucket_base == 0.
    dims_big = Dimensions(65536, 4096, 8192, 32, 1, 8)
    found_192 = False
    for vmem_mb in range(4, 64):
        vmem_lim = vmem_mb * 1024 * 1024
        try:
            base_192 = calculate_tiling(dims_big, lhs, rhs, vmem_lim, "silu")
        except ValueError:
            continue
        if base_192.tile_m == 192:
            found_192 = True
            t_192 = batch_invariant_tiling(dims_big, lhs, rhs, vmem_lim, "silu")
            assert t_192.bucket_base == 128
            assert t_192.tile_m == 128
            assert t_192.tile_m % t_192.bucket_base == 0
            break
    assert found_192, "Expected at least one VMEM budget yielding base tile_m == 192"


def test_muse_moe_impl_env_choices_and_unset_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_torchtpu.layers.adapter.custom_ops.muse_moe_tpu import (
        MuseFusedMoeArgs,
        muse_moe_impl,
    )
    from vllm_torchtpu.layers.adapter.quantization.unquantized import (
        VllmUnquantizedFusedMoEMethod,
    )

    monkeypatch.setenv("VLLM_TPU_MUSE_MOE_IMPL", "EAGER")
    assert muse_moe_impl() == "eager"

    monkeypatch.setenv("VLLM_TPU_MUSE_MOE_IMPL", "bogus")
    with pytest.raises(ValueError, match="Invalid value 'bogus'"):
        envs.environment_variables["VLLM_TPU_MUSE_MOE_IMPL"]()
    monkeypatch.setenv("VLLM_TPU_MUSE_MOE_IMPL", "fused")

    # Building MuseFusedMoeArgs before process_weights_after_loading sets
    # _tpu_activation_str must fail loudly rather than silently assuming silu.
    unset_method = object.__new__(VllmUnquantizedFusedMoEMethod)
    unset_method._tpu_activation_str = None
    fake_routed = SimpleNamespace(
        quant_method=unset_method,
        w13_weight=torch.zeros((1, 8, 16)),
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=False)),
        _experts_start=None,
    )
    fake_moe = SimpleNamespace(experts=SimpleNamespace(routed_experts=fake_routed))
    with pytest.raises(RuntimeError, match="_tpu_activation_str"):
        MuseFusedMoeArgs(fake_moe)

    # Empty (0-element) routed expert weights cleanly disable the per-layer
    # fused op without raising.
    offloaded_routed = SimpleNamespace(
        quant_method=unset_method,
        w13_weight=torch.empty(0),
        moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(use_ep=True)),
    )
    offloaded_args = MuseFusedMoeArgs(
        SimpleNamespace(experts=SimpleNamespace(routed_experts=offloaded_routed))
    )
    assert offloaded_args.op is None
    assert "routed expert weights are empty" in offloaded_args.unsupported


def test_fixed_order_sum_is_sequential() -> None:
    chunks = torch.tensor([[1e8], [1.0], [-1e8], [1.0]], dtype=torch.float32)
    # Left to right in fp32: ((1e8 + 1) - 1e8) + 1 = 1, unlike a pairwise tree (2).
    assert det_collectives._fixed_order_sum(chunks).item() == 1.0


def test_pad_lm_head_weight() -> None:
    w = torch.randn(25256, 16, dtype=torch.bfloat16)
    p = bi.pad_lm_head_weight(w)
    assert p.shape == (25344, 16)
    assert torch.equal(p[:25256], w) and not p[25256:].any()
    w2 = torch.randn(256, 16)
    assert bi.pad_lm_head_weight(w2) is w2


def test_det_projection_falls_back_off_tpu() -> None:
    class Lin(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 8))
            self.bias = None

        def forward(self, x: torch.Tensor):
            return x @ self.weight.t(), None

    lin = Lin()
    x = torch.randn(3, 8)
    torch.testing.assert_close(bi.det_projection(lin, x), x @ lin.weight.t())


def test_det_rms_normalize_rows_independent_of_batch() -> None:
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal((32, 256)), jnp.float32)
    ref = np.asarray(x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-5))
    full = np.asarray(det_rms.det_rms_normalize(x, eps=1e-5, tile_m=8, interpret=True))
    np.testing.assert_allclose(full, ref, rtol=1e-6, atol=1e-6)
    for sub in (x[:8], x[8:24], x):
        got = np.asarray(
            det_rms.det_rms_normalize(sub, eps=1e-5, tile_m=8, interpret=True)
        )
        start = int(np.argmax(np.all(np.asarray(x) == np.asarray(sub[:1]), axis=1)))
        np.testing.assert_array_equal(got, full[start : start + sub.shape[0]])


def test_rms_normalize_falls_back_off_tpu() -> None:
    x = torch.randn(5, 64, dtype=torch.float32)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(bi.rms_normalize(x, 1e-5), ref)


def test_muse_moe_jax_pre_projected_matches_inline() -> None:
    from vllm_torchtpu.layers.adapter.custom_ops.muse_moe_tpu import _muse_moe_jax

    rng = np.random.default_rng(2)
    h = jnp.asarray(rng.standard_normal((4, 16)), jnp.float32)
    rw = jnp.asarray(rng.standard_normal((16, 4)), jnp.float32)
    rb = jnp.zeros((4,), jnp.float32)
    pw = jnp.asarray(rng.standard_normal((16, 8)), jnp.float32)
    pn = jnp.zeros((8,), jnp.float32)
    w13 = jnp.asarray(rng.standard_normal((4, 8, 16)), jnp.float32)
    w2 = jnp.asarray(rng.standard_normal((4, 8, 8)), jnp.float32)
    post_n = jnp.zeros((8,), jnp.float32)
    start = jnp.array([0], jnp.int32)
    inline = _muse_moe_jax(
        h, rw, rb, pw, pn, w13, None, w2, None, post_n, start, topk=2
    )
    z_pre = jnp.dot(h, pw, preferred_element_type=jnp.float32)
    pre_proj = _muse_moe_jax(
        h,
        rw,
        rb,
        z_pre,
        pn,
        w13,
        None,
        w2,
        None,
        post_n,
        start,
        topk=2,
        pre_projected=True,
    )
    np.testing.assert_array_equal(np.asarray(inline), np.asarray(pre_proj))


def test_hd64_rpa_wrapper_and_entry_threading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_torchtpu.layers.adapter import attention as attention_adapter
    from vllm_torchtpu.layers.core import attention_interface
    from vllm_torchtpu.models.vllm.muse_spark import attention_compat as ac

    seen: dict[str, object] = {}

    def fake_inner(kv, q, k, v, seq_lens, bt, q_loc, dist, sink):
        seen["hd"] = q.shape[-1]
        seen["dist"] = dist.tolist()
        return q + 1.0

    wrapped = ac.wrap_hd64_rpa_kernel(fake_inner)
    q = torch.zeros(16, 4, 128)
    k = torch.zeros(16, 2, 128)
    v = torch.zeros(16, 2, 128)
    dist = torch.tensor([2, 2, 5], dtype=torch.int32)
    out = wrapped(None, q, k, v, None, None, None, dist, None)
    assert out.shape == (16, 4, 64)
    assert seen == {"hd": 64, "dist": [0, 0, 5]}

    # Verify _muse_rpa_kernel_entry passes hd64_rpa_func explicitly without
    # mutating attention_interface.ragged_paged_attention_hd64.
    orig_global = attention_interface.ragged_paged_attention_hd64
    captured: dict[str, object] = {}

    def fake_impl(*args: object, **kwargs: object):
        assert attention_interface.ragged_paged_attention_hd64 is orig_global
        captured["hd64_rpa_func"] = kwargs.get("hd64_rpa_func")
        return args[0], args[1]

    monkeypatch.setattr(attention_adapter, "_pallas_rpa_kernel_impl", fake_impl)
    dummy = jnp.zeros((1,), jnp.float32)
    dummy_mesh = Mesh(np.asarray(jax.devices()[:1]), ("model",))
    ac._muse_rpa_kernel_entry(
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        None,
        None,
        None,
        None,
        mesh=dummy_mesh,
        sliding_window=None,
        skip_kv_update=False,
    )
    assert captured["hd64_rpa_func"] is ac._hd64_with_prefill_block
    assert attention_interface.ragged_paged_attention_hd64 is orig_global


def test_s2_fused_rms_norm_and_residual_gate_kernels() -> None:
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal((16, 128)), jnp.bfloat16)
    g = jnp.asarray(rng.standard_normal((16, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((128,)) * 0.1, jnp.bfloat16)
    res = jnp.asarray(rng.standard_normal((16, 128)), jnp.float32)
    alpha = jnp.asarray(rng.uniform(0.5, 1.0, (128,)), jnp.float32)
    beta = jnp.asarray(rng.uniform(0.1, 0.5, (128,)), jnp.float32)
    eps = 1e-5
    scale = 5.4730064863875

    xf = x.astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + jnp.float32(eps))
    norm_bf16 = (xf * inv).astype(jnp.bfloat16)

    got_s = det_rms.det_rms_norm_scaled(
        x, eps=eps, scale=scale, out_dtype=jnp.bfloat16, tile_m=8, interpret=True
    )
    ref_s = (norm_bf16.astype(jnp.float32) * jnp.float32(scale)).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got_s), np.asarray(ref_s))

    got_og = det_rms.det_o_norm_sigmoid_gate(x, g, eps=eps, tile_m=8, interpret=True)
    ref_og = norm_bf16 * jax.nn.sigmoid(g.astype(jnp.float32)).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got_og), np.asarray(ref_og))

    got_c = det_rms.det_centered_rms_norm(
        res, w, eps=eps, out_dtype=jnp.bfloat16, tile_m=8, interpret=True
    )
    inv_r = jax.lax.rsqrt(
        jnp.mean(res * res, axis=-1, keepdims=True) + jnp.float32(eps)
    )
    ref_c = (res * inv_r * (1.0 + w.astype(jnp.float32))).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got_c), np.asarray(ref_c))

    got_is = det_rms.det_input_scaled_rms_norm(
        x, w, eps=eps, out_dtype=jnp.bfloat16, tile_m=8, interpret=True
    )
    sc = (xf * (1.0 + w.astype(jnp.float32))).astype(jnp.bfloat16).astype(jnp.float32)
    inv_sc = jax.lax.rsqrt(jnp.mean(sc * sc, axis=-1, keepdims=True) + jnp.float32(eps))
    ref_is = (sc * inv_sc).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(got_is), np.asarray(ref_is))

    got_rg = det_rms.det_rms_norm_residual_gate(
        res, x, alpha, beta, eps=eps, out_dtype=jnp.float32, tile_m=8, interpret=True
    )
    ref_rg = alpha * res + beta * norm_bf16.astype(jnp.float32)
    np.testing.assert_allclose(
        np.asarray(got_rg), np.asarray(ref_rg), rtol=1e-6, atol=1e-6
    )
    sub_rg = det_rms.det_rms_norm_residual_gate(
        res[:8],
        x[:8],
        alpha,
        beta,
        eps=eps,
        out_dtype=jnp.float32,
        tile_m=8,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(sub_rg), np.asarray(got_rg)[:8])

    got_is_rg = det_rms.det_input_scaled_rms_norm_residual_gate(
        res, x, w, alpha, beta, eps=eps, out_dtype=jnp.float32, tile_m=8, interpret=True
    )
    ref_is_rg = alpha * res + beta * ref_is.astype(jnp.float32)
    np.testing.assert_allclose(
        np.asarray(got_is_rg), np.asarray(ref_is_rg), rtol=1e-6, atol=1e-6
    )
    sub_is_rg = det_rms.det_input_scaled_rms_norm_residual_gate(
        res[:8],
        x[:8],
        w,
        alpha,
        beta,
        eps=eps,
        out_dtype=jnp.float32,
        tile_m=8,
        interpret=True,
    )
    np.testing.assert_array_equal(np.asarray(sub_is_rg), np.asarray(got_is_rg)[:8])


@pytest.mark.parametrize("n_valid", [0, 5, 8, 13, 24])
def test_masked_input_scaled_norm_matches_unmasked(n_valid: int) -> None:
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.standard_normal((24, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((128,)) * 0.1, jnp.bfloat16)
    full = np.asarray(
        det_rms.det_input_scaled_rms_norm(
            x, w, eps=1e-8, out_dtype=jnp.bfloat16, tile_m=8, interpret=True
        )
    )
    got = np.asarray(
        det_rms.det_input_scaled_rms_norm_masked(
            x,
            w,
            jnp.int32(n_valid),
            eps=1e-8,
            out_dtype=jnp.bfloat16,
            tile_m=8,
            interpret=True,
        )
    )
    np.testing.assert_array_equal(got[:n_valid], full[:n_valid])


def test_prefill_queries_per_block() -> None:
    from vllm_torchtpu.models.vllm.muse_spark import attention_compat as ac

    assert [ac.prefill_queries_per_block(t, 32) for t in (32, 1024, 2048, 8192)] == [
        32,
        32,
        64,
        128,
    ]
    assert ac.prefill_queries_per_block(2048, 128) == 128


@pytest.mark.tpu_test
def test_det_kernels_on_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    from vllm_torchtpu.layers.core.fused_moe_gmm import gmm_wrapper

    rng = np.random.default_rng(42)
    # K=4096 > default_tile_k(4096)==2048 exercises multi-tile K accumulation on TPU.
    x = jnp.asarray(rng.standard_normal((32, 4096)) * 0.25, jnp.bfloat16)
    w = jnp.asarray(rng.standard_normal((4096, 512)) * 0.25, jnp.bfloat16)
    ref_mm = np.asarray(
        jnp.dot(x, w, preferred_element_type=jnp.float32), dtype=np.float32
    )
    for rhs_nk in (False, True):
        wl = w.T if rhs_nk else w
        full = np.asarray(det_mm.det_matmul(x, wl, rhs_nk=rhs_nk), dtype=np.float32)
        sub = np.asarray(
            det_mm.det_matmul(x[4:12], wl, rhs_nk=rhs_nk), dtype=np.float32
        )
        np.testing.assert_allclose(full, ref_mm, rtol=1e-2, atol=2e-2)
        np.testing.assert_array_equal(sub, full[4:12])

    # Exercise RMS, residual-gate, and masked scalar-prefetch kernels vs JAX refs.
    x_n = jnp.asarray(rng.standard_normal((32, 256)), jnp.bfloat16)
    g_n = jnp.asarray(rng.standard_normal((32, 256)), jnp.bfloat16)
    w_n = jnp.asarray(rng.standard_normal((256,)) * 0.1, jnp.bfloat16)
    res_n = jnp.asarray(rng.standard_normal((32, 256)), jnp.float32)
    alpha = jnp.asarray(rng.uniform(0.5, 1.0, (256,)), jnp.float32)
    beta = jnp.asarray(rng.uniform(0.1, 0.5, (256,)), jnp.float32)
    eps = 1e-5
    scale = 5.4730064863875

    xf = x_n.astype(jnp.float32)
    ref_rms = np.asarray(
        xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + jnp.float32(eps))
    )
    full_rms = np.asarray(det_rms.det_rms_normalize(xf, eps=eps))
    sub_rms = np.asarray(det_rms.det_rms_normalize(xf[4:12], eps=eps))
    np.testing.assert_allclose(full_rms, ref_rms, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(sub_rms, full_rms[4:12])

    norm_bf16 = jnp.asarray(ref_rms, jnp.bfloat16)
    ref_s = np.asarray(
        (norm_bf16.astype(jnp.float32) * jnp.float32(scale)).astype(jnp.bfloat16),
        np.float32,
    )
    full_s = np.asarray(
        det_rms.det_rms_norm_scaled(x_n, eps=eps, scale=scale, out_dtype=jnp.bfloat16),
        np.float32,
    )
    sub_s = np.asarray(
        det_rms.det_rms_norm_scaled(
            x_n[4:12], eps=eps, scale=scale, out_dtype=jnp.bfloat16
        ),
        np.float32,
    )
    np.testing.assert_allclose(full_s, ref_s, rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(sub_s, full_s[4:12])

    ref_og = np.asarray(
        norm_bf16 * jax.nn.sigmoid(g_n.astype(jnp.float32)).astype(jnp.bfloat16),
        np.float32,
    )
    full_og = np.asarray(det_rms.det_o_norm_sigmoid_gate(x_n, g_n, eps=eps), np.float32)
    sub_og = np.asarray(
        det_rms.det_o_norm_sigmoid_gate(x_n[4:12], g_n[4:12], eps=eps), np.float32
    )
    np.testing.assert_allclose(full_og, ref_og, rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(sub_og, full_og[4:12])

    inv_r = jax.lax.rsqrt(
        jnp.mean(res_n * res_n, axis=-1, keepdims=True) + jnp.float32(eps)
    )
    ref_c = np.asarray(
        (res_n * inv_r * (1.0 + w_n.astype(jnp.float32))).astype(jnp.bfloat16),
        np.float32,
    )
    full_c = np.asarray(
        det_rms.det_centered_rms_norm(res_n, w_n, eps=eps, out_dtype=jnp.bfloat16),
        np.float32,
    )
    sub_c = np.asarray(
        det_rms.det_centered_rms_norm(
            res_n[4:12], w_n, eps=eps, out_dtype=jnp.bfloat16
        ),
        np.float32,
    )
    np.testing.assert_allclose(full_c, ref_c, rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(sub_c, full_c[4:12])

    sc = (xf * (1.0 + w_n.astype(jnp.float32))).astype(jnp.bfloat16).astype(jnp.float32)
    ref_is_j = (
        sc * jax.lax.rsqrt(jnp.mean(sc * sc, axis=-1, keepdims=True) + jnp.float32(eps))
    ).astype(jnp.bfloat16)
    full_is = np.asarray(
        det_rms.det_input_scaled_rms_norm(x_n, w_n, eps=eps, out_dtype=jnp.bfloat16),
        np.float32,
    )
    sub_is = np.asarray(
        det_rms.det_input_scaled_rms_norm(
            x_n[4:12], w_n, eps=eps, out_dtype=jnp.bfloat16
        ),
        np.float32,
    )
    np.testing.assert_allclose(
        full_is, np.asarray(ref_is_j, np.float32), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_array_equal(sub_is, full_is[4:12])

    for n_valid in (0, 5, 13, 24):
        got_m = np.asarray(
            det_rms.det_input_scaled_rms_norm_masked(
                x_n[:24],
                w_n,
                jnp.int32(n_valid),
                eps=eps,
                out_dtype=jnp.bfloat16,
            ),
            np.float32,
        )
        np.testing.assert_array_equal(got_m[:n_valid], full_is[:n_valid])

    ref_rg = np.asarray(alpha * res_n + beta * norm_bf16.astype(jnp.float32))
    full_rg = np.asarray(
        det_rms.det_rms_norm_residual_gate(
            res_n, x_n, alpha, beta, eps=eps, out_dtype=jnp.float32
        )
    )
    sub_rg = np.asarray(
        det_rms.det_rms_norm_residual_gate(
            res_n[4:12], x_n[4:12], alpha, beta, eps=eps, out_dtype=jnp.float32
        )
    )
    np.testing.assert_allclose(full_rg, ref_rg, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(sub_rg, full_rg[4:12])

    ref_is_rg = np.asarray(alpha * res_n + beta * ref_is_j.astype(jnp.float32))
    full_is_rg = np.asarray(
        det_rms.det_input_scaled_rms_norm_residual_gate(
            res_n, x_n, w_n, alpha, beta, eps=eps, out_dtype=jnp.float32
        )
    )
    sub_is_rg = np.asarray(
        det_rms.det_input_scaled_rms_norm_residual_gate(
            res_n[4:12], x_n[4:12], w_n, alpha, beta, eps=eps, out_dtype=jnp.float32
        )
    )
    np.testing.assert_allclose(full_is_rg, ref_is_rg, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(sub_is_rg, full_is_rg[4:12])

    # Exercise gmm_wrapper(..., tile_info=batch_invariant_tiling) vs ref across M.
    lhs_g = jnp.asarray(rng.standard_normal((32, 1024)) * 0.25, jnp.bfloat16)
    rhs_g = jnp.asarray(rng.standard_normal((2, 1024, 256)) * 0.25, jnp.bfloat16)
    gs_full = jnp.asarray([16, 16], dtype=jnp.int32)
    gs_sub = jnp.asarray([8, 0], dtype=jnp.int32)
    go = jnp.asarray([0], dtype=jnp.int32)
    out_g_full = np.asarray(
        gmm_wrapper(
            lhs_g,
            rhs_g,
            None,
            None,
            gs_full,
            go,
            preferred_element_type=jnp.bfloat16,
            tile_info=batch_invariant_tiling,
        ),
        np.float32,
    )
    out_g_sub = np.asarray(
        gmm_wrapper(
            lhs_g[:8],
            rhs_g,
            None,
            None,
            gs_sub,
            go,
            preferred_element_type=jnp.bfloat16,
            tile_info=batch_invariant_tiling,
        ),
        np.float32,
    )
    ref_g0 = np.asarray(
        jnp.dot(lhs_g[:16], rhs_g[0], preferred_element_type=jnp.float32).astype(
            jnp.bfloat16
        ),
        np.float32,
    )
    np.testing.assert_allclose(out_g_full[:16], ref_g0, rtol=2e-2, atol=3e-2)
    np.testing.assert_array_equal(out_g_sub[:8], out_g_full[:8])


@pytest.mark.multichip
def test_det_collectives_sharded_ops_on_multichip() -> None:
    if jax.default_backend() != "tpu" or len(jax.devices()) < 2:
        pytest.skip("Requires at least 2 TPU TensorCores")
    world = 2
    mesh = Mesh(np.asarray(jax.devices()[:world]), (det_collectives._AXIS,))
    spec = P(det_collectives._AXIS, None)
    rng = np.random.default_rng(99)

    # 1. Verify _det_reduce_scatter and _sharded_mean_sq against psum references
    #    and verify bit-exact row invariance across m=8 vs m=24.
    part_24 = rng.standard_normal((world, 24, 512)).astype(np.float32) * 0.5
    part_24_bf16 = jnp.asarray(part_24, jnp.bfloat16)
    rs_fn = jax.jit(
        jax.shard_map(
            lambda p: (
                det_collectives._det_reduce_scatter(p, world),
                det_collectives._sharded_mean_sq(
                    p[:, : 512 // world].astype(jnp.float32), world
                ),
            ),
            mesh=mesh,
            in_specs=spec,
            out_specs=(spec, spec),
            check_vma=False,
        )
    )
    got_rs_24, got_msq_24 = rs_fn(
        jax.device_put(
            part_24_bf16.reshape(world * 24, 512),
            jax.sharding.NamedSharding(mesh, spec),
        )
    )
    got_rs_8, got_msq_8 = rs_fn(
        jax.device_put(
            part_24_bf16[:, :8].reshape(world * 8, 512),
            jax.sharding.NamedSharding(mesh, spec),
        )
    )
    rs_24_np = np.asarray(got_rs_24, np.float32).reshape(world, 24, 256)
    rs_8_np = np.asarray(got_rs_8, np.float32).reshape(world, 8, 256)
    ref_sum = np.sum(np.asarray(part_24_bf16, np.float32), axis=0)
    np.testing.assert_allclose(rs_24_np[0], ref_sum[:, :256], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rs_24_np[1], ref_sum[:, 256:], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(rs_8_np, rs_24_np[:, :8])

    sharded_full = np.concatenate(
        [
            np.asarray(part_24_bf16[0, :, :256], np.float32),
            np.asarray(part_24_bf16[1, :, :256], np.float32),
        ],
        axis=-1,
    )
    ref_msq = np.mean(sharded_full * sharded_full, axis=-1, keepdims=True)
    msq_24_np = np.asarray(got_msq_24, np.float32).reshape(world, 24, 1)
    msq_8_np = np.asarray(got_msq_8, np.float32).reshape(world, 8, 1)
    np.testing.assert_allclose(msq_24_np[0], ref_msq, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(msq_8_np, msq_24_np[:, :8])

    # 2. Exercise _build_ar_op, _get_attn_post_op (with non-16 local heads and
    #    reversed z_perm), _get_moe_post_op, and _get_dense_post_op on hardware
    #    against NumPy references and across m=8 vs m=24.
    saved_mesh_fn = det_collectives._tp_mesh
    saved_order_fn = det_collectives.tp_collective_order
    det_collectives._ATTN_POST_OPS.clear()
    det_collectives._MOE_POST_OPS.clear()
    det_collectives._DENSE_POST_OPS.clear()
    det_collectives._AR_OPS.clear()
    try:
        det_collectives._tp_mesh = lambda: mesh  # type: ignore[assignment]
        det_collectives.tp_collective_order = lambda: (1, 0)  # type: ignore[assignment]
        sharding = jax.sharding.NamedSharding(mesh, spec)

        # 2a. _build_ar_op
        ar_op = det_collectives._build_ar_op(torch.bfloat16, torch.float32)
        assert ar_op is not None
        ar_jit = ar_op._init_fn.jit_fn
        x_ar = jnp.asarray(rng.standard_normal((world, 24, 256)) * 0.5, jnp.bfloat16)
        got_ar_24 = np.asarray(
            ar_jit(jax.device_put(x_ar.reshape(world * 24, 256), sharding)),
            np.float32,
        ).reshape(world, 24, 256)
        got_ar_8 = np.asarray(
            ar_jit(jax.device_put(x_ar[:, :8].reshape(world * 8, 256), sharding)),
            np.float32,
        ).reshape(world, 8, 256)
        ref_ar = np.sum(np.asarray(x_ar, np.float32), axis=0)
        np.testing.assert_allclose(got_ar_24[0], ref_ar, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(got_ar_24[1], ref_ar, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(got_ar_8, got_ar_24[:, :8])

        # 2b. _get_attn_post_op with 8 local heads (width 512) + pre_exp + z_perm=(1,0)
        attn_op = det_collectives._get_attn_post_op(
            1e-5, 1e-8, 1e-5, False, False, True, False, head_dim=64
        )
        assert attn_op is not None
        attn_jit = attn_op._init_fn.jit_fn
        h_dim = 512
        hs = h_dim // world
        ao_a = jnp.asarray(rng.standard_normal((world, 24, 512)) * 0.25, jnp.bfloat16)
        h_a = jnp.asarray(rng.standard_normal((24, h_dim)) * 0.25, jnp.bfloat16)
        h_a_rep = jnp.broadcast_to(h_a[None], (world, 24, h_dim))
        gw_a = jnp.asarray(rng.standard_normal((world, h_dim, 512)) * 0.2, jnp.bfloat16)
        ow_a = jnp.asarray(rng.standard_normal((world, 512, h_dim)) * 0.2, jnp.bfloat16)
        rs_a = jnp.asarray(rng.standard_normal((world, 24, hs)) * 0.25, jnp.float32)
        as_a = jnp.asarray(rng.uniform(0.6, 0.9, (world, 1, hs)), jnp.float32)
        bs_a = jnp.asarray(rng.uniform(0.1, 0.4, (world, 1, hs)), jnp.float32)
        ws_a = jnp.asarray(rng.standard_normal((world, 1, hs)) * 0.05, jnp.bfloat16)
        pew_a = jnp.asarray(
            rng.standard_normal((world, h_dim, 128)) * 0.2, jnp.bfloat16
        )

        def _put_sharded(arr: jax.Array, rows: int | None = None) -> jax.Array:
            a = arr[:, :rows] if (rows is not None and arr.shape[1] == 24) else arr
            return jax.device_put(
                a.reshape((world * a.shape[1], *a.shape[2:])), sharding
            )

        args_24 = [
            _put_sharded(t)
            for t in (ao_a, h_a_rep, gw_a, ow_a, rs_a, as_a, bs_a, ws_a, pew_a)
        ]
        args_8 = [
            _put_sharded(t, rows=8)
            for t in (ao_a, h_a_rep, gw_a, ow_a, rs_a, as_a, bs_a, ws_a, pew_a)
        ]
        hid24, hffn24, zpre24 = attn_jit(*args_24)
        hid8, hffn8, zpre8 = attn_jit(*args_8)
        np.testing.assert_array_equal(
            np.asarray(hid8).reshape(world, 8, hs),
            np.asarray(hid24).reshape(world, 24, hs)[:, :8],
        )
        np.testing.assert_array_equal(
            np.asarray(hffn8, np.float32).reshape(world, 8, h_dim),
            np.asarray(hffn24, np.float32).reshape(world, 24, h_dim)[:, :8],
        )
        zpre24_np = np.asarray(zpre24, np.float32).reshape(world, 24, 256)
        zpre8_np = np.asarray(zpre8, np.float32).reshape(world, 8, 256)
        np.testing.assert_array_equal(zpre8_np, zpre24_np[:, :8])
        # Verify z_perm=(1, 0) puts rank 1's block at [:128] and rank 0's at [128:].
        hffn0 = np.asarray(hffn24, np.float32).reshape(world, 24, h_dim)[0]
        ref_z_rank1 = hffn0 @ np.asarray(pew_a[1], np.float32)
        ref_z_rank0 = hffn0 @ np.asarray(pew_a[0], np.float32)
        np.testing.assert_allclose(
            zpre24_np[0, :, :128], ref_z_rank1, rtol=2e-2, atol=3e-2
        )
        np.testing.assert_allclose(
            zpre24_np[0, :, 128:], ref_z_rank0, rtol=2e-2, atol=3e-2
        )

        # 2c. _get_moe_post_op & _get_dense_post_op row invariance + reference check
        moe_op = det_collectives._get_moe_post_op(1e-8, 1e-5, False, True, False)
        dense_op = det_collectives._get_dense_post_op(1e-8, 1e-5, True, False)
        assert moe_op is not None and dense_op is not None
        mfp = jnp.asarray(rng.standard_normal((world, 24, 256)) * 0.25, jnp.float32)
        pw_m = jnp.asarray(rng.standard_normal((world, 256, hs)) * 0.2, jnp.bfloat16)
        pnw_s = jnp.asarray(rng.standard_normal((world, 1, hs)) * 0.05, jnp.bfloat16)
        nw_s = jnp.asarray(rng.standard_normal((world, 1, hs)) * 0.05, jnp.bfloat16)
        m_hid24, m_hn24 = moe_op._init_fn.jit_fn(
            *[_put_sharded(t) for t in (mfp, pw_m, rs_a, pnw_s, as_a, bs_a, nw_s)]
        )
        m_hid8, m_hn8 = moe_op._init_fn.jit_fn(
            *[
                _put_sharded(t, rows=8)
                for t in (mfp, pw_m, rs_a, pnw_s, as_a, bs_a, nw_s)
            ]
        )
        np.testing.assert_array_equal(
            np.asarray(m_hid8).reshape(world, 8, hs),
            np.asarray(m_hid24).reshape(world, 24, hs)[:, :8],
        )
        np.testing.assert_array_equal(
            np.asarray(m_hn8, np.float32).reshape(world, 8, h_dim),
            np.asarray(m_hn24, np.float32).reshape(world, 24, h_dim)[:, :8],
        )

        ffs = jnp.asarray(rng.standard_normal((world, 24, hs)) * 0.25, jnp.bfloat16)
        d_hid24, d_hn24 = dense_op._init_fn.jit_fn(
            *[_put_sharded(t) for t in (ffs, rs_a, pnw_s, as_a, bs_a, nw_s)]
        )
        d_hid8, d_hn8 = dense_op._init_fn.jit_fn(
            *[_put_sharded(t, rows=8) for t in (ffs, rs_a, pnw_s, as_a, bs_a, nw_s)]
        )
        np.testing.assert_array_equal(
            np.asarray(d_hid8).reshape(world, 8, hs),
            np.asarray(d_hid24).reshape(world, 24, hs)[:, :8],
        )
        np.testing.assert_array_equal(
            np.asarray(d_hn8, np.float32).reshape(world, 8, h_dim),
            np.asarray(d_hn24, np.float32).reshape(world, 24, h_dim)[:, :8],
        )
    finally:
        det_collectives._tp_mesh = saved_mesh_fn  # type: ignore[assignment]
        det_collectives.tp_collective_order = saved_order_fn  # type: ignore[assignment]


def test_det_collectives_caching_and_head_dim_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify tp_position_block op caching, dynamic head_dim, and FakeTensor shapes."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch_tpu._internal.pallas import pallas as pallas_impl

    devices = [
        pallas_impl._PallasDevice(id=i, process_index=i, device_kind="TPU v7")
        for i in range(2)
    ]
    mesh = Mesh(np.asarray(devices), (det_collectives._AXIS,))
    monkeypatch.setattr(det_collectives, "_tp_mesh", lambda: mesh)
    monkeypatch.setattr(det_collectives, "tp_collective_order", lambda: (1, 0))
    monkeypatch.setattr(det_collectives, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        det_collectives, "get_tensor_model_parallel_world_size", lambda: 2
    )
    det_collectives._TP_GATHER_OPS.clear()
    det_collectives._ATTN_POST_OPS.clear()

    with FakeTensorMode() as fm:
        w1 = fm.from_tensor(torch.zeros((16, 128), dtype=torch.bfloat16))
        w2 = fm.from_tensor(torch.zeros((16, 128), dtype=torch.bfloat16))
        w3 = fm.from_tensor(torch.zeros((16, 128), dtype=torch.float32))
        # Calling tp_position_block twice on the same shape+dtype must reuse the
        # cached op rather than failing on duplicate custom_op registration.
        b1 = det_collectives.tp_position_block(w1, dim=0)
        b2 = det_collectives.tp_position_block(w2, dim=0)
        b3 = det_collectives.tp_position_block(w3, dim=0)
        assert b1.shape == (16, 128) and b1.dtype == torch.bfloat16
        assert b2.shape == (16, 128) and b2.dtype == torch.bfloat16
        assert b3.shape == (16, 128) and b3.dtype == torch.float32
        assert len(det_collectives._TP_GATHER_OPS) == 2

        # Verify det_attn_post_sandwich supports non-16 local head counts (e.g. 32 heads
        # of head_dim=64 -> width 2048 at TP=4) and rejects unaligned widths.
        ao = fm.from_tensor(torch.zeros((4, 2048), dtype=torch.bfloat16))
        h_in = fm.from_tensor(torch.zeros((4, 1024), dtype=torch.bfloat16))
        gw = fm.from_tensor(torch.zeros((1024, 2048), dtype=torch.bfloat16))
        ow = fm.from_tensor(torch.zeros((2048, 1024), dtype=torch.bfloat16))
        rs = fm.from_tensor(torch.zeros((4, 512), dtype=torch.float32))
        a_s = fm.from_tensor(torch.zeros((512,), dtype=torch.float32))
        b_s = fm.from_tensor(torch.zeros((512,), dtype=torch.float32))
        w_s = fm.from_tensor(torch.zeros((512,), dtype=torch.bfloat16))
        hid_s, h_ffn = det_collectives.det_attn_post_sandwich(
            ao,
            h_in,
            gw,
            ow,
            rs,
            a_s,
            b_s,
            w_s,
            o_eps=1e-5,
            post_attn_eps=1e-8,
            pre_ffn_eps=1e-5,
            gate_rhs_nk=False,
            o_rhs_nk=False,
            head_dim=64,
        )
        assert hid_s.shape == (4, 512) and hid_s.dtype == torch.float32
        assert h_ffn.shape == (4, 1024) and h_ffn.dtype == torch.bfloat16


def test_det_collectives_fullgraph_dynamo_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _tp_mesh() and _build_ar_op trace under torch.compile(fullgraph=True)."""
    from torch_tpu._internal.pallas import pallas as pallas_impl

    devices = [
        pallas_impl._PallasDevice(id=i, process_index=i, device_kind="TPU v7")
        for i in range(2)
    ]
    mesh = Mesh(np.asarray(devices), (det_collectives._AXIS,))
    tp_stub = SimpleNamespace(world_size=2)
    monkeypatch.setattr(det_collectives, "get_tp_group", lambda: tp_stub)
    monkeypatch.setattr(
        det_collectives, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setitem(det_collectives._TP_MESH_CACHE, 2, mesh)
    monkeypatch.setitem(det_collectives._TP_ORDER_CACHE, 2, (0, 1))
    det_collectives._AR_OPS.clear()
    det_collectives._ATTN_POST_OPS.clear()
    det_collectives._MOE_POST_OPS.clear()
    det_collectives._DENSE_POST_OPS.clear()
    det_collectives._build_ar_op(torch.bfloat16, torch.bfloat16)
    det_collectives._get_attn_post_op(1e-5, 1e-8, 1e-5, False, False, False, False, 64)
    det_collectives._get_moe_post_op(1e-8, 1e-5, False, True, False)
    det_collectives._get_dense_post_op(1e-8, 1e-5, True, False)

    # Swap cached custom ops for tensor ops so CPU execution succeeds while
    # exercising every _get_*_op -> _tp_mesh() -> mesh.shape lookup inside Dynamo.
    det_collectives._AR_OPS[(2, torch.bfloat16, torch.bfloat16)] = lambda t: t + 1
    det_collectives._ATTN_POST_OPS[
        (2, 64, 1e-5, 1e-8, 1e-5, False, False, False, False)
    ] = lambda *a: a[0] + 1
    det_collectives._MOE_POST_OPS[(2, 1e-8, 1e-5, False, True, False)] = (
        lambda *a: a[0] + 1
    )
    det_collectives._DENSE_POST_OPS[(2, 1e-8, 1e-5, True, False)] = lambda *a: a[0] + 1

    def _fn(x: torch.Tensor) -> torch.Tensor:
        ar = det_collectives._build_ar_op(x.dtype, x.dtype)
        ap = det_collectives._get_attn_post_op(
            1e-5, 1e-8, 1e-5, False, False, False, False, 64
        )
        mp = det_collectives._get_moe_post_op(1e-8, 1e-5, False, True, False)
        dp = det_collectives._get_dense_post_op(1e-8, 1e-5, True, False)
        assert ar is not None and ap is not None and mp is not None and dp is not None
        return dp(mp(ap(ar(x))))

    compiled = torch.compile(_fn, backend="eager", fullgraph=True)
    inp = torch.zeros((4, 256), dtype=torch.bfloat16)
    out = compiled(inp)
    assert out.shape == (4, 256) and out.dtype == torch.bfloat16
    assert torch.all(out == 4)
