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
"""kernels/collectives/residual_ar: its exchange layout, and both kernels on 8
CPU devices in interpret mode with the race detector (a subprocess, since the
device count is fixed when JAX starts), against the unfused math."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.collectives import residual_ar

N_RANKS = 8
HIDDEN = 6144
# qkv_proj's per-rank output width.
QKV = 1536
# MiniMax-M3.1 decode rows: c4 b1, c16 b3 (both stack column halves), b4.
ROWS = (8, 24, 32)
EPS = 1e-6


@pytest.mark.parametrize(
    "m, want", [(8, (16, 3072)), (24, (48, 3072)), (32, (32, 6144))]
)
def test_exchange_shape(m, want):
    assert residual_ar._exchange_shape(m, HIDDEN) == want


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu_info.get_tpu_info = tpu7x
    k = residual_ar
    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    rng = np.random.default_rng(0)
    nw = jnp.asarray(0.1 * rng.standard_normal((HIDDEN,)), jnp.bfloat16)
    wo = jnp.asarray(0.05 * rng.standard_normal((1024, HIDDEN)), jnp.float8_e4m3fn)
    so = jnp.asarray(2.0 ** rng.integers(-3, 3, (HIDDEN,)), jnp.bfloat16)
    wq = jnp.asarray(0.05 * rng.standard_normal((HIDDEN, QKV)), jnp.float8_e4m3fn)
    sq = jnp.asarray(2.0 ** rng.integers(-3, 3, (QKV,)), jnp.bfloat16)

    def body(x, res, a, rank):
        n_, r_ = k.ar_add_norm(x, res, nw, rank, N_RANKS, EPS)
        q, qs, rq = k.ar_add_norm(x, res, nw, rank, N_RANKS, EPS, quant=True)
        o, r2 = k.matmul_ar_norm(a, wo, so, res, nw, rank, N_RANKS, EPS)
        xr = jax.lax.psum(x.astype(jnp.float32), "d").astype(jnp.bfloat16)
        n0, r0 = k.reference_add_norm(xr, res, nw, EPS)
        q0, qs0 = k.quantize_rows(n0)
        pr = jax.lax.psum(k.reference_matmul(a, wo, so).astype(jnp.float32), "d")
        o0, r20 = k.reference_add_norm(pr.astype(jnp.bfloat16), res, nw, EPS)
        y, ry = k.ar_add_norm_matmul(x, res, nw, wq, sq, rank, N_RANKS, EPS)
        yq, ryq = k.ar_add_norm_matmul(
            x, res, nw, wq, sq, rank, N_RANKS, EPS, quant=True
        )
        y0 = k.reference_matmul(n0, wq, sq)
        yq0 = k.e4m3_rows_matmul(q0, qs0, wq, sq)
        f32 = jnp.float32
        return [v.astype(f32) for v in (n_, n0, r_, r0, q, q0, qs, qs0, rq, r0)] + [
            v.astype(f32) for v in (o, o0, r2, r20, y, y0, ry, yq, yq0, ryq)
        ]

    spec = P("d")
    run = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 4, out_specs=spec, check_vma=False
        )
    )
    put = lambda v: jax.device_put(v, NamedSharding(mesh, spec))  # noqa: E731
    result = {}
    for m in ROWS:
        # Partials whose sums are exact in bf16, so the all-reduce's order
        # cannot change r; one residual on every rank.
        x = (
            (
                np.arange(N_RANKS)[:, None, None]
                + np.arange(m)[None, :, None] * 3
                + np.arange(HIDDEN)[None, None, :] // 128
            )
            % 7
            - 3
        ).astype(np.float32)
        res = np.tile(rng.standard_normal((m, HIDDEN)), (N_RANKS, 1))
        a = rng.standard_normal((N_RANKS * m, 1024))
        ranks = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
        pltpu.reset_tpu_interpret_mode_state()
        outs = run(
            put(jnp.asarray(x.reshape(-1, HIDDEN), jnp.bfloat16)),
            put(jnp.asarray(res, jnp.bfloat16)),
            put(jnp.asarray(a, jnp.bfloat16)),
            put(ranks),
        )
        n_, n0, r_, r0, q, q0, qs, qs0, rq, _, o, o0, r2, r20 = map(
            np.asarray, outs[:14]
        )
        y, y0, ry, yq, yq0, ryq = map(np.asarray, outs[14:])
        result[str(m)] = dict(
            normed=bool(np.array_equal(n_, n0)),
            residual=bool(np.array_equal(r_, r0) and np.array_equal(rq, r0)),
            e4m3=bool(np.array_equal(q, q0) and np.array_equal(qs, qs0)),
            # o_proj's K order differs from XLA's dot: bf16 roundings.
            o_proj=float(np.abs(o - o0).max() / np.abs(o0).max()),
            o_proj_residual=float(np.abs(r2 - r20).max() / np.abs(r20).max()),
            # qkv_proj's K order differs from XLA's dot too.
            qkv=float(np.abs(y - y0).max() / np.abs(y0).max()),
            qkv_e4m3=float(np.abs(yq - yq0).max() / np.abs(yq0).max()),
            qkv_residual=bool(np.array_equal(ry, r0) and np.array_equal(ryq, r0)),
            races=bool(interpret_pallas_call.races.races_found),
        )
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}",
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=900
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("m", ROWS)
def test_kernels_on_8_devices(on_8_devices, m):
    case = on_8_devices[str(m)]
    assert case["normed"] and case["residual"] and case["e4m3"], case
    assert case["o_proj"] < 2**-6 and case["o_proj_residual"] < 2**-6, case
    assert case["qkv"] < 2**-6 and case["qkv_e4m3"] < 2**-6, case
    assert case["qkv_residual"], case
    assert not case["races"], case


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
