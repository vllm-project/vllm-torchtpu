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
"""Tile-selection, gating, and TPU parity tests for `kernels.rmsnorm_fp8`."""

from unittest import mock

import jax
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels import rmsnorm_fp8 as kernel
from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)


def _reference_rmsnorm_fp8_matmul(
    x: jax.Array,
    gamma: jax.Array,
    w_q: jax.Array,
    w_s: jax.Array,
    eps: float = 1e-6,
    gamma_offset: float = 0.0,
    *,
    quantize_activation: bool = True,
) -> tuple[jax.Array, jax.Array]:
    xf = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(xf), axis=-1, keepdims=True)
    gf = gamma.astype(jnp.float32) + gamma_offset
    norm_bf16 = (xf * jax.lax.rsqrt(var + eps) * gf).astype(x.dtype)
    return (
        xla_quantized_matmul(
            norm_bf16, w_q, w_s, quantize_activation=quantize_activation
        ),
        norm_bf16,
    )


def _fp32_oracle(norm_bf16: jax.Array, w_q: jax.Array, w_s: jax.Array) -> jax.Array:
    w_deq = w_q.astype(jnp.float32) * w_s[None, :]
    return (norm_bf16.astype(jnp.float32) @ w_deq).astype(norm_bf16.dtype)


def _assert_kernel_quant_matches_unfused(
    x: jax.Array,
    gamma: jax.Array,
    norm_bf16: jax.Array,
    *,
    gamma_offset: float,
    bm: int,
    vmem: int,
) -> None:
    """Runs the Pallas kernel itself and checks it against the unfused path.

    The reference is ``quantize_tensor`` of the unfused bf16 norm, which is what
    ``xla_quantized_matmul`` feeds to the matmul. The kernel sums squares in a
    different order, so a few normalized values land on the other side of a
    bf16 rounding boundary and their FP8 value moves by one step.
    """
    x_q, x_s = jax.jit(
        lambda xi, gi: kernel.fused_rmsnorm_fp8_quant(
            xi,
            gi,
            jnp.float8_e4m3fn,
            eps=1e-6,
            gamma_offset=gamma_offset,
            bm=bm,
            vmem_capacity_bytes=vmem,
        )
    )(x, gamma)
    ref_q, ref_s = jax.jit(lambda nb: quantize_tensor(nb, jnp.float8_e4m3fn))(norm_bf16)
    assert x_q.shape == ref_q.shape and x_q.dtype == ref_q.dtype
    assert x_s.shape == ref_s.shape == (x.shape[0], 1) and x_s.dtype == jnp.float32

    # A row scale moves only if the row's abs-max element crosses a rounding
    # boundary, and then by at most one bf16 step (2**-7 relative plus f32
    # rounding from /448).
    s, rs = np.asarray(x_s), np.asarray(ref_s)
    assert float(np.mean(s == rs)) >= 0.99
    assert float(np.max(np.abs(s - rs) / rs)) <= 2.0**-7 + 1e-6

    q = np.asarray(x_q.astype(jnp.float32))
    rq = np.asarray(ref_q.astype(jnp.float32))
    differs = q != rq
    assert float(np.mean(differs)) < 0.05
    # FP8 E4M3 has 3 mantissa bits, so one step is at most 1/8 of the larger
    # magnitude (2**-9 between subnormals).
    qd, rqd = q[differs], rq[differs]
    one_step = np.maximum(np.abs(qd), np.abs(rqd)) / 8 + 2.0**-9
    assert np.all(np.abs(qd - rqd) <= one_step)


def test_select_block_m_and_should_use_pallas_gating_cpu():
    """CPU check of VMEM tile bounds, bucket gating, admission errors, and W8A16."""
    v7x_vmem = int(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1).vmem_capacity_bytes
    )
    assert kernel.vmem_limit_bytes(v7x_vmem) == 56 * 1024 * 1024

    # k=5376 (Gemma 4 31B): bm=512 fits within 56 MiB (42.5 MiB with f32 scratch).
    assert kernel.select_block_m(128, 5376, v7x_vmem) == 128
    assert kernel.select_block_m(256, 5376, v7x_vmem) == 256
    assert kernel.select_block_m(512, 5376, v7x_vmem) == 512
    assert kernel.select_block_m(16384, 5376, v7x_vmem) == 512

    # k=8192 (70B hidden size): bm=512 exceeds 56 MiB on multi-tile -> steps to 256.
    assert kernel.select_block_m(16384, 8192, v7x_vmem) == 256

    # Alignment and empty/invalid bounds (bm must be 32-aligned):
    assert kernel.select_block_m(0, 5376, v7x_vmem) is None
    assert kernel.select_block_m(216, 5376, v7x_vmem) is None
    assert kernel.select_block_m(128, 64, v7x_vmem) is None
    assert kernel.select_block_m(128, 130, v7x_vmem) is None
    assert kernel.select_block_m(130, 5376, v7x_vmem) is None
    assert kernel.select_block_m(600, 5376, v7x_vmem) is None
    assert kernel.select_block_m(256, 5376, 0) is None

    for n_out, m, w8a16, expected in [
        (8192, 64, 0, False),  # below FP8_RMSNORM_MIN_TOKENS=512
        (8192, 128, 0, False),
        (8192, 216, 0, False),
        (8192, 256, 0, False),
        (8192, 512, 0, True),
        (8192, 512, 512, False),  # W8A16 bucket stays on native XLA W8A16
        (8192, 1024, 512, True),  # above w8a16_max_rows -> Pallas FP8
        (21504, 16384, 0, True),  # TP=2 gate_up_proj full prefill
        (10752, 16384, 0, True),  # TP=4 gate_up_proj full prefill
        (5376, 4096, 0, True),  # TP=8 gate_up_proj 4K prefill
        (5376, 8192, 0, True),  # TP=8 gate_up_proj 8K prefill
        (5376, 16384, 0, False),  # TP=8 narrow gate_up_proj (N < 6144) at 16K prefill
        (8192, 514, 0, False),  # M % 32 != 0
        (0, 512, 0, False),  # invalid N <= 0
    ]:
        assert (
            kernel.should_use_pallas_rmsnorm_fp8(
                m, 5376, n_out, v7x_vmem, w8a16_max_rows=w8a16
            )
            is expected
        )

    # Exact admission checks on kernel entry points:
    x_f32 = jnp.ones((128, 128), dtype=jnp.float32)
    x_bf16 = jnp.ones((128, 128), dtype=jnp.bfloat16)
    g_bf16 = jnp.ones((128,), dtype=jnp.bfloat16)
    wq_fp8 = jnp.ones((128, 256), dtype=jnp.float8_e4m3fn)
    ws_f32 = jnp.ones((256,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="2D bfloat16"):
        kernel.fused_rmsnorm_fp8_matmul(
            x_f32, g_bf16, wq_fp8, ws_f32, vmem_capacity_bytes=v7x_vmem
        )
    with pytest.raises(ValueError, match="gamma of shape"):
        kernel.fused_rmsnorm_fp8_matmul(
            x_bf16,
            jnp.ones((64,), dtype=jnp.bfloat16),
            wq_fp8,
            ws_f32,
            vmem_capacity_bytes=v7x_vmem,
        )
    with pytest.raises(ValueError, match="1-byte floating FP8"):
        kernel.fused_rmsnorm_fp8_matmul(
            x_bf16,
            g_bf16,
            jnp.ones((128, 256), dtype=jnp.int8),
            ws_f32,
            vmem_capacity_bytes=v7x_vmem,
        )
    with pytest.raises(ValueError, match="vmem_capacity_bytes must be >= 0"):
        kernel.fused_rmsnorm_fp8_matmul(
            x_bf16, g_bf16, wq_fp8, ws_f32, vmem_capacity_bytes=-1
        )
    with pytest.raises(ValueError, match="Invalid tile config"):
        kernel.fused_rmsnorm_fp8_quant(
            x_bf16,
            g_bf16,
            jnp.float8_e4m3fn,
            bm=48,
            vmem_capacity_bytes=v7x_vmem,
        )
    # m % bm == 0 holds; only bm % 32 == 0 rejects both multi-tile and single-tile.
    for x_bad, bad_bm in (
        (x_bf16, 16),
        (jnp.ones((216, 128), dtype=jnp.bfloat16), 216),
    ):
        with pytest.raises(ValueError, match="Invalid tile config"):
            kernel.fused_rmsnorm_fp8_quant(
                x_bad,
                g_bf16,
                jnp.float8_e4m3fn,
                bm=bad_bm,
                vmem_capacity_bytes=v7x_vmem,
            )

    # CPU JAX check of fallback paths (m < 512 W8A8 and m >= 512 W8A16):
    k1, k2, k3 = jax.random.split(jax.random.key(42), 3)
    x_small = jax.random.normal(k1, (64, 128), jnp.float32).astype(jnp.bfloat16)
    gamma = (1.0 + 0.2 * jax.random.normal(k2, (128,), jnp.float32)).astype(
        jnp.bfloat16
    )
    w_raw = jax.random.normal(k3, (128, 256), jnp.float32) * 0.1
    wq, ws = quantize_tensor(w_raw.T, jnp.float8_e4m3fn)
    wq, ws = wq.T, ws.reshape(-1)

    out_m64 = kernel.fused_rmsnorm_fp8_matmul(
        x_small, gamma, wq, ws, eps=1e-6, gamma_offset=0.0, vmem_capacity_bytes=v7x_vmem
    )
    ref_m64, _ = _reference_rmsnorm_fp8_matmul(
        x_small, gamma, wq, ws, eps=1e-6, gamma_offset=0.0, quantize_activation=True
    )
    np.testing.assert_array_equal(np.asarray(out_m64), np.asarray(ref_m64))

    # m=512 is >= FP8_RMSNORM_MIN_TOKENS, so this specifically verifies that
    # fused_rmsnorm_fp8_matmul forwards w8a16_max_rows to the Pallas gate.
    x_512 = jax.random.normal(k1, (512, 128), jnp.float32).astype(jnp.bfloat16)
    out_w8a16 = kernel.fused_rmsnorm_fp8_matmul(
        x_512,
        gamma,
        wq,
        ws,
        eps=1e-6,
        gamma_offset=1.0,
        w8a16_max_rows=512,
        vmem_capacity_bytes=v7x_vmem,
    )
    ref_w8a16, _ = _reference_rmsnorm_fp8_matmul(
        x_512, gamma, wq, ws, eps=1e-6, gamma_offset=1.0, quantize_activation=False
    )
    ref_w8a8, _ = _reference_rmsnorm_fp8_matmul(
        x_512, gamma, wq, ws, eps=1e-6, gamma_offset=1.0, quantize_activation=True
    )
    np.testing.assert_array_equal(np.asarray(out_w8a16), np.asarray(ref_w8a16))
    # Must-fail control: W8A16 skips activation quantization and differs from W8A8.
    assert not np.array_equal(np.asarray(out_w8a16), np.asarray(ref_w8a8))


@pytest.mark.tpu_test
# m=256 (below FP8_RMSNORM_MIN_TOKENS=512) and (m=16384, k=1024, n=512) (narrow N
# above _NARROW_N_PREFILL_MAX_M=8192) exercise XLA fallback; other m>=512 cases
# exercise Pallas (m=256, k=1024 also tests fused_rmsnorm_fp8_quant on a zero row).
@pytest.mark.parametrize("m", [256, 512, 1024, 4096, 16384])
@pytest.mark.parametrize("k,n", [(5376, 8192), (1024, 512)])
@pytest.mark.parametrize("gamma_offset", [0.0, 1.0])
def test_pallas_rmsnorm_fp8_parity_tpu(m, k, n, gamma_offset):
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")

    vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    bm = kernel.select_block_m(m, k, vmem)
    if bm is None:
        pytest.skip("shape not eligible")

    k1, k2, k3 = jax.random.split(jax.random.key(m + k + int(gamma_offset * 10)), 3)
    x = jax.random.normal(k1, (m, k), jnp.float32).astype(jnp.bfloat16)
    gamma_dtype = jnp.float32 if gamma_offset == 1.0 else jnp.bfloat16
    gamma = (0.3 * jax.random.normal(k2, (k,), jnp.float32)).astype(gamma_dtype)
    if gamma_offset == 0.0:
        gamma = (1.0 + gamma.astype(jnp.float32)).astype(jnp.bfloat16)
    w = jax.random.normal(k3, (k, n), jnp.float32) * 0.02
    w_q, w_s = quantize_tensor(w.T, jnp.float8_e4m3fn)
    w_q, w_s = w_q.T, w_s.reshape(-1)

    if m == 256 and k == 1024:
        x_zero_row = x.at[3].set(0)
        xq_z, xs_z = jax.jit(
            lambda xi, gi: kernel.fused_rmsnorm_fp8_quant(
                xi,
                gi,
                jnp.float8_e4m3fn,
                eps=1e-6,
                gamma_offset=gamma_offset,
                bm=bm,
                vmem_capacity_bytes=vmem,
            )
        )(x_zero_row, gamma)
        assert float(xs_z[3, 0]) == 1.0
        assert not bool(jnp.any(jnp.isnan(xq_z.astype(jnp.float32))))
        assert bool(jnp.all(xq_z[3].astype(jnp.float32) == 0.0))

    uses_pallas = kernel.should_use_pallas_rmsnorm_fp8(m, k, n, vmem)
    with mock.patch.object(
        kernel, "fused_rmsnorm_fp8_quant", wraps=kernel.fused_rmsnorm_fp8_quant
    ) as quant_spy:
        fused = jax.jit(
            lambda xi, gi, wq, ws: kernel.fused_rmsnorm_fp8_matmul(
                xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
            )
        )(x, gamma, w_q, w_s)
    # The entry point traces the Pallas kernel exactly when the bucket is
    # admitted; a silent fallback to the XLA path would leave this at 0.
    assert quant_spy.call_count == int(uses_pallas)
    ref, norm_bf16 = jax.jit(
        lambda xi, gi, wq, ws: _reference_rmsnorm_fp8_matmul(
            xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
        )
    )(x, gamma, w_q, w_s)
    oracle = jax.jit(_fp32_oracle)(norm_bf16, w_q, w_s)

    assert fused.shape == ref.shape and fused.dtype == ref.dtype
    f = np.asarray(fused.astype(jnp.float32))
    r = np.asarray(ref.astype(jnp.float32))
    o = np.asarray(oracle.astype(jnp.float32))

    cos_fo = float(np.sum(f * o) / (np.linalg.norm(f) * np.linalg.norm(o)))
    cos_ro = float(np.sum(r * o) / (np.linalg.norm(r) * np.linalg.norm(o)))
    assert cos_fo > 0.9995, f"fused vs fp32 oracle cosine too low: {cos_fo:.6f}"

    if uses_pallas:
        _assert_kernel_quant_matches_unfused(
            x, gamma, norm_bf16, gamma_offset=gamma_offset, bm=bm, vmem=vmem
        )
        cos_fr = float(np.sum(f * r) / (np.linalg.norm(f) * np.linalg.norm(r)))
        rms_rel_fr = float(np.linalg.norm(f - r) / np.linalg.norm(r))
        assert cos_fr > 0.9999, f"fused vs ref cosine too low: {cos_fr:.6f}"
        assert rms_rel_fr < 0.005, (
            f"fused vs ref RMS rel diff too high: {rms_rel_fr:.6f}"
        )
        assert abs(cos_fo - cos_ro) < 2e-4
        # Must-fail control: perturbing fused output trips the RMS check.
        f_bad = f.copy()
        f_bad[0] += 10.0
        assert float(np.linalg.norm(f_bad - r) / np.linalg.norm(r)) >= 0.005
    else:
        ref_inlined = jax.jit(
            lambda xi, gi, wq, ws: _reference_rmsnorm_fp8_matmul(
                xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
            )[0]
        )(x, gamma, w_q, w_s)
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(ref_inlined))


@pytest.mark.tpu_test
def test_pallas_rmsnorm_fp8_per_tensor_scale_and_bf16_gemma_gamma_tpu():
    """Per-tensor (1,) weight scale with a bf16 gamma and gamma_offset=1.0."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")

    m, k, n, gamma_offset = 512, 1024, 512, 1.0
    vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes)
    bm = kernel.select_block_m(m, k, vmem)
    assert bm is not None and kernel.should_use_pallas_rmsnorm_fp8(m, k, n, vmem)

    k1, k2, k3 = jax.random.split(jax.random.key(7), 3)
    x = jax.random.normal(k1, (m, k), jnp.float32).astype(jnp.bfloat16)
    gamma = (0.3 * jax.random.normal(k2, (k,), jnp.float32)).astype(jnp.bfloat16)
    w = jax.random.normal(k3, (k, n), jnp.float32) * 0.02
    w_q, w_s = quantize_tensor(w.reshape(1, -1), jnp.float8_e4m3fn)
    w_q, w_s = w_q.reshape(k, n), w_s.reshape(-1)
    assert w_s.shape == (1,)

    fused = jax.jit(
        lambda xi, gi, wq, ws: kernel.fused_rmsnorm_fp8_matmul(
            xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
        )
    )(x, gamma, w_q, w_s)
    ref, norm_bf16 = jax.jit(
        lambda xi, gi, wq, ws: _reference_rmsnorm_fp8_matmul(
            xi, gi, wq, ws, eps=1e-6, gamma_offset=gamma_offset
        )
    )(x, gamma, w_q, w_s)

    _assert_kernel_quant_matches_unfused(
        x, gamma, norm_bf16, gamma_offset=gamma_offset, bm=bm, vmem=vmem
    )
    assert fused.shape == ref.shape == (m, n) and fused.dtype == ref.dtype
    f = np.asarray(fused.astype(jnp.float32))
    r = np.asarray(ref.astype(jnp.float32))
    cos_fr = float(np.sum(f * r) / (np.linalg.norm(f) * np.linalg.norm(r)))
    rms_rel_fr = float(np.linalg.norm(f - r) / np.linalg.norm(r))
    assert cos_fr > 0.9999, f"fused vs ref cosine too low: {cos_fr:.6f}"
    assert rms_rel_fr < 0.005, f"fused vs ref RMS rel diff too high: {rms_rel_fr:.6f}"
