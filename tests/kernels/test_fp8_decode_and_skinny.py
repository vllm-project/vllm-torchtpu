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
"""Pallas interpret, TPU, and contract tests for FP8 decode/skinny & collectives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu import tpu_info
from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce
from vllm_torchtpu.kernels.quantized_matmul import (
    decode_gemv,
    fp8_oproj_all_reduce,
    fp8_skinny,
    fp8_window_all_gather,
    util,
)


def test_is_tpu_v7x_returns_false_on_cpu_backend(monkeypatch) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU-backend fallback check")
    if not hasattr(tpu_info.envs, "CPU_ONLY"):
        monkeypatch.setattr(tpu_info.envs, "CPU_ONLY", False, raising=False)
    assert tpu_info.is_tpu_v7x() is False


def test_decode_gemv_fp8_and_prequantized_and_nt_match_reference() -> None:
    assert decode_gemv.supports(16, 256, 512, jnp.float8_e4m3fn, 1)
    assert decode_gemv.supports(8, 256, 512, jnp.float8_e4m3fn, 1, row_multiple=8)
    assert not decode_gemv.supports(8, 256, 512, jnp.float8_e4m3fn, 2)
    assert decode_gemv.supports_nt(16, 256, 512, itemsize=2)

    m, k, n = 16, 256, 512
    k1, k2, k3 = jax.random.split(jax.random.key(301), 3)
    x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
    w_f32 = jax.random.normal(k2, (n, k), dtype=jnp.float32) * 0.25
    w_n_k, w_scale = util.quantize_tensor(w_f32, jnp.float8_e4m3fn, dim=1)
    w_k_n = w_n_k.T
    s_1d = w_scale.squeeze(-1)

    y_gemv = decode_gemv.fp8_decode_matmul(
        x, w_k_n, s_1d, tile_bytes=65536, interpret=True
    )
    y_ref = util.xla_quantized_matmul(x, w_k_n, s_1d)
    np.testing.assert_allclose(
        np.asarray(y_gemv, dtype=np.float32),
        np.asarray(y_ref, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )

    x_q, x_scale = util.quantize_tensor(x, jnp.float8_e4m3fn, dim=1)
    y_pre_pallas = decode_gemv.fp8_decode_prequantized_matmul(
        x_q,
        x_scale,
        w_k_n,
        s_1d,
        out_dtype=jnp.bfloat16,
        tile_bytes=65536,
        interpret=True,
    )
    y_pre_xla = util.xla_prequantized_matmul(
        x_q, x_scale, w_k_n, s_1d, out_dtype=jnp.bfloat16
    )
    np.testing.assert_allclose(
        np.asarray(y_pre_pallas, dtype=np.float32),
        np.asarray(y_pre_xla, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )

    w_bf16 = jax.random.normal(k3, (n, k), dtype=jnp.bfloat16) * 0.2
    y_nt = decode_gemv.decode_matmul_nt(x, w_bf16, tile_bytes=65536, interpret=True)
    y_nt_ref = jnp.dot(x, w_bf16.T, preferred_element_type=jnp.float32).astype(
        jnp.bfloat16
    )
    np.testing.assert_allclose(
        np.asarray(y_nt, dtype=np.float32),
        np.asarray(y_nt_ref, dtype=np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_fp8_skinny_tuned_shapes_and_pallas_interpret() -> None:
    expected_tuned = {(6144, 2688), (2048, 16384), (1024, 6144), (6144, 1536)}
    assert set(fp8_skinny.TUNED) == expected_tuned
    assert (2048, 8192) not in fp8_skinny.TUNED
    assert fp8_skinny.scale_dma_applies(1024, 6144)
    assert not fp8_skinny.scale_dma_applies(2048, 16384)

    m, k, n_log = 8, 6144, 2624
    n_pad = fp8_skinny.padded_n(n_log)
    assert n_pad == 2688
    k1, k2 = jax.random.split(jax.random.key(515))
    x = jax.random.normal(k1, (m, k), dtype=jnp.bfloat16)
    w_f32 = jax.random.normal(k2, (n_pad, k), dtype=jnp.float32) * 0.1
    w_n_k, s_n = util.quantize_tensor(w_f32, jnp.float8_e4m3fn, dim=1)
    w_k_n = w_n_k.T
    s_1d = s_n.squeeze(-1).astype(jnp.float32)

    assert fp8_skinny.supports(m, w_k_n, s_1d)
    assert fp8_skinny.supports(4, w_k_n, s_1d)
    assert not fp8_skinny.supports(0, w_k_n, s_1d)
    assert not fp8_skinny.supports(fp8_skinny.MAX_M + 8, w_k_n, s_1d)

    cfg = fp8_skinny.pick_config(k, n_pad)
    y = fp8_skinny.fp8_skinny_matmul(
        x, w_k_n, s_1d, n_logical=n_log, config=cfg, interpret=True
    )
    y_ref = util.xla_quantized_matmul(x, w_k_n[:, :n_log], s_1d[:n_log])

    assert y.shape == (m, n_log)
    y_f32 = np.asarray(y, dtype=np.float32)
    y_ref_f32 = np.asarray(y_ref, dtype=np.float32)
    rel_l2 = float(np.linalg.norm(y_f32 - y_ref_f32) / np.linalg.norm(y_ref_f32))
    assert rel_l2 < 1e-2
    atol = 2e-2 if jax.default_backend() == "cpu" else 2e-1
    np.testing.assert_allclose(y_f32, y_ref_f32, rtol=2e-2, atol=atol)


@pytest.mark.parametrize("num_cores", [8, 16])
def test_collective_geometry_and_fused_oproj_contracts(num_cores: int) -> None:
    assert fp8_window_all_gather.supports(8, 2048, 16384, num_cores)
    assert not fp8_window_all_gather.supports(0, 2048, 16384, num_cores)
    assert not fp8_window_all_gather.supports(32, 2048, 16384, num_cores)

    assert fp8_oproj_all_reduce.supports(8, 1024, 6144, num_cores)
    assert not fp8_oproj_all_reduce.supports(0, 1024, 6144, num_cores)
    assert fp8_oproj_all_reduce.one_row_supported(6144)
    waves = fp8_oproj_all_reduce.owner_waves(16, 6144, tile=2048)
    assert len(waves) == 3 and sum(len(w) for w in waves) == 16

    assert all_reduce.form_for(16, 6144, 2) in ("oneshot", "twoshot")
    assert all_reduce.fold_geometry(4, 6144, num_cores=num_cores, tile_rows=8) is None
    geom = all_reduce.fold_geometry(16, 6144, num_cores=num_cores, tile_rows=8)
    assert geom is not None
    blocks = [
        jnp.full((8, 256), float(r + 1), dtype=jnp.bfloat16) for r in range(num_cores)
    ]
    ref = all_reduce.reference(jnp.stack(blocks, axis=0))
    expected = sum(range(1, num_cores + 1))
    np.testing.assert_allclose(
        np.asarray(ref, dtype=np.float32), float(expected), rtol=1e-2
    )


@pytest.mark.tpu_test
def test_decode_gemv_and_fp8_skinny_on_tpu() -> None:
    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU backend")
    k1, k2 = jax.random.split(jax.random.key(909))
    x = jax.random.normal(k1, (16, 6144), dtype=jnp.bfloat16)
    w_n_k, s_n = util.quantize_tensor(
        jax.random.normal(k2, (2688, 6144), dtype=jnp.float32) * 0.1,
        jnp.float8_e4m3fn,
        dim=1,
    )
    w_k_n, s_1d = w_n_k.T, s_n.squeeze(-1).astype(jnp.float32)
    y_ref = np.asarray(util.xla_quantized_matmul(x, w_k_n, s_1d), np.float32)
    for fn in (decode_gemv.fp8_decode_matmul, fp8_skinny.fp8_skinny_matmul):
        y = np.asarray(fn(x, w_k_n, s_1d), np.float32)
        np.testing.assert_allclose(y, y_ref, rtol=2e-2, atol=2e-1)


@pytest.mark.tpu_test
@pytest.mark.multichip
def test_small_collectives_and_fused_oproj_on_multichip_tpu() -> None:
    if jax.default_backend() != "tpu" or len(jax.devices("tpu")) < 8:
        pytest.skip("Requires >= 8 TPU devices")
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices("tpu")[:8]), ("tp",))
    x = jax.random.normal(jax.random.key(71), (8, 16, 2048), dtype=jnp.bfloat16)
    p3, p2 = P("tp", None, None), P("tp", None)
    got_ar = jax.shard_map(
        lambda s: all_reduce.small_all_reduce(s[0], num_cores=8, axis="tp")[None],
        mesh=mesh,
        in_specs=p3,
        out_specs=p3,
        check_vma=False,
    )(x)
    ref_ar = np.asarray(all_reduce.reference(x), np.float32)
    for r in range(8):
        np.testing.assert_allclose(
            np.asarray(got_ar[r], np.float32), ref_ar, rtol=2e-2, atol=5e-2
        )

    k1, k2 = jax.random.split(jax.random.key(72))
    xo = jax.random.normal(k1, (8, 8, 1024), dtype=jnp.bfloat16) * 0.2
    w_nk, s_n = util.quantize_tensor(
        jax.random.normal(k2, (8 * 1024, 6144), dtype=jnp.float32).T * 0.1,
        jnp.float8_e4m3fn,
        dim=1,
    )
    w_sh, s_rep = (
        w_nk.T.reshape(8, 1024, 6144),
        jnp.broadcast_to(s_n.squeeze(-1).astype(jnp.float32), (8, 6144)),
    )
    ranks = jnp.arange(8, dtype=jnp.int32)[:, None]
    got_op = jax.shard_map(
        lambda xs, ws, ss, rk: fp8_oproj_all_reduce.fp8_matmul_all_reduce(
            xs[0], ws[0], ss[0], rk[0], num_cores=8, axis="tp"
        )[None],
        mesh=mesh,
        in_specs=(p3, p3, p2, p2),
        out_specs=p3,
        check_vma=False,
    )(xo, w_sh, s_rep, ranks)
    ref_op = sum(
        util.xla_quantized_matmul(xo[r], w_sh[r], s_rep[0]).astype(jnp.float32)
        for r in range(8)
    )
    np.testing.assert_allclose(
        np.asarray(got_op[0], np.float32),
        np.asarray(ref_op, np.float32),
        rtol=3e-2,
        atol=3e-1,
    )
