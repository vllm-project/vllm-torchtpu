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

"""The q_b column window fused with the all-gather of the windows
against the replicated fp8_skinny kernel, on 16 CPU
devices in TPU interpret mode with the race detector (a subprocess: the host
device count must be set before JAX starts). It covers the DMA and semaphore
protocol and the arithmetic, not Mosaic lowering or timing."""

import contextlib
import io
import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_window_all_gather as wag

N_RANKS = 16
ROWS = (4, 8, 16)


def test_supports_the_q_b_windows():
    assert wag.supports(4, 2048, 16384, 16)
    assert wag.supports(8, 2048, 16384, 16)
    assert wag.supports(16, 2048, 16384, 16)
    assert not wag.supports(17, 2048, 16384, 16)  # above the skinny rows
    assert not wag.supports(4, 2048, 16384, 32)  # 512-column windows, 1024 tiles
    assert not wag.supports(4, 1024, 6144, 16)  # o_proj tiles N by 2048


def test_should_reject_invalid_d2d_pair_first_core_count():
    with pytest.raises(ValueError, match="power-of-2 num_cores >= 4"):
        wag.fp8_window_all_gather(
            jnp.zeros((8, 2048), dtype=jnp.bfloat16),
            jnp.zeros((2048, 16384), dtype=jnp.float8_e4m3fn),
            jnp.ones((16384,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            num_cores=2,
            axis="d",
            d2d_pair_first=True,
        )


def test_should_gate_qnorm_fusion_to_tiny_decode_buckets():
    assert not wag.supports_qnorm(0, 2048, 16384, 16)
    assert wag.supports_qnorm(1, 2048, 16384, 16)
    assert wag.supports_qnorm(4, 2048, 16384, 16)
    assert not wag.supports_qnorm(5, 2048, 16384, 16)
    assert not wag.supports_qnorm(8, 2048, 16384, 16)


def test_should_reject_mismatched_norm_weight_shape():
    with pytest.raises(ValueError, match="norm_w must have shape"):
        wag.fp8_window_all_gather(
            jnp.zeros((4, 2048), dtype=jnp.bfloat16),
            jnp.zeros((2048, 16384), dtype=jnp.float8_e4m3fn),
            jnp.ones((16384,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.int32),
            num_cores=16,
            axis="d",
            norm_w=jnp.ones((1024,), dtype=jnp.bfloat16),
        )


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu7x
    tpu_info.get_tpu_info = tpu7x
    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    k, n = 2048, 16384
    result = {}
    for m in ROWS:
        k1, k2, k3 = jax.random.split(jax.random.key(m), 3)
        x = (jax.random.normal(k1, (m, k)) * 2).astype(jnp.bfloat16)
        w = (jax.random.normal(k2, (k, n)) * 0.5).astype(jnp.float8_e4m3fn)
        ws = jax.random.uniform(k3, (n,), jnp.float32, 0.5, 1.5)
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
            ref = np.asarray(fp8_skinny.fp8_skinny_matmul(x, w, ws)).view(np.uint16)
        for lanes in (False, True):
            if lanes:
                mesh = Mesh(np.asarray(devices).reshape(1, N_RANKS), ("lane", "d"))
                spec = P(("lane", "d"))
            else:
                mesh = Mesh(np.asarray(devices), ("d",))
                spec = P("d")

            def body(x, w, ws, r, lane, lanes=lanes):
                return wag.fp8_window_all_gather(
                    x,
                    w,
                    ws,
                    r.reshape(1),
                    num_cores=N_RANKS,
                    axis="d",
                    lane_axis="lane" if lanes else None,
                    lane=lane.reshape(1) if lanes else None,
                )

            fn = jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(spec,) * 5,
                    out_specs=spec,
                    check_vma=False,
                )
            )
            args = (
                jnp.tile(x, (N_RANKS, 1)),
                jnp.tile(w, (N_RANKS, 1)),
                jnp.tile(ws, (N_RANKS,)),
                jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1),
                jnp.zeros((N_RANKS, 1), jnp.int32),
            )
            with pltpu.force_tpu_interpret_mode(
                pltpu.InterpretParams(detect_races=True)
            ):
                pltpu.reset_tpu_interpret_mode_state()
                with contextlib.redirect_stdout(io.StringIO()):
                    out = np.asarray(jax.block_until_ready(fn(*args)))
                races = bool(interpret_pallas_call.races.races_found)
            out = out.reshape(N_RANKS, m, n).view(np.uint16)
            result[f"{m}:{int(lanes)}"] = bool(
                not races and all(np.array_equal(out[r], ref) for r in range(N_RANKS))
            )
    # Verify fused RMSNorm matches independent f32-intermediate norm bitwise
    # and differs from bf16-truncated normalization (discriminating oracle).
    rn_m = wag.QNORM_MAX_M
    rk1, rk2, rk3, rk4 = jax.random.split(jax.random.key(99), 4)
    rx = (jax.random.normal(rk1, (rn_m, k)) * 2.5 + 0.1).astype(jnp.bfloat16)
    rw = (jax.random.normal(rk2, (k, n)) * 0.5).astype(jnp.float8_e4m3fn)
    rws = jax.random.uniform(rk3, (n,), jnp.float32, 0.5, 1.5)
    rnw = jax.random.uniform(rk4, (k,), jnp.float32, 0.3, 1.7).astype(jnp.bfloat16)
    eps = 1e-6
    rxf = rx.astype(jnp.float32)
    rinv = jax.lax.rsqrt(jnp.mean(rxf * rxf, axis=-1, keepdims=True) + jnp.float32(eps))
    x_norm_f32 = ((rxf * rinv) * rnw.astype(jnp.float32)).astype(jnp.bfloat16)
    x_norm_trunc = ((rxf * rinv).astype(jnp.bfloat16) * rnw).astype(jnp.bfloat16)
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()):
        ref_rn = np.asarray(fp8_skinny.fp8_skinny_matmul(x_norm_f32, rw, rws)).view(
            np.uint16
        )
        ref_trunc = np.asarray(
            fp8_skinny.fp8_skinny_matmul(x_norm_trunc, rw, rws)
        ).view(np.uint16)
    mesh_rn = Mesh(np.asarray(devices), ("d",))

    def body_rn(x, w, ws, r, nw):
        return wag.fp8_window_all_gather(
            x,
            w,
            ws,
            r.reshape(1),
            num_cores=N_RANKS,
            axis="d",
            norm_w=nw,
            eps=eps,
        )

    fn_rn = jax.jit(
        jax.shard_map(
            body_rn,
            mesh=mesh_rn,
            in_specs=(P("d"),) * 5,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    args_rn = (
        jnp.tile(rx, (N_RANKS, 1)),
        jnp.tile(rw, (N_RANKS, 1)),
        jnp.tile(rws, (N_RANKS,)),
        jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1),
        jnp.tile(rnw, (N_RANKS,)),
    )
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out_rn = np.asarray(jax.block_until_ready(fn_rn(*args_rn)))
        rn_races = bool(interpret_pallas_call.races.races_found)
    out_rn = out_rn.reshape(N_RANKS, rn_m, n).view(np.uint16)
    result["rn_exact"] = bool(
        not rn_races and all(np.array_equal(out_rn[r], ref_rn) for r in range(N_RANKS))
    )
    result["rn_distinguishes_trunc"] = bool(not np.array_equal(ref_rn, ref_trunc))
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_16_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("lanes", [False, True])
@pytest.mark.parametrize("rows", ROWS)
def test_gathered_windows_are_the_replicated_projection(on_16_devices, rows, lanes):
    """Every rank's output is bitwise fp8_skinny's on the whole weight, on a 1-D
    mesh and on a (lane, DCP) mesh with one lane; no races."""
    assert on_16_devices[f"{rows}:{int(lanes)}"]


def test_should_match_f32_rmsnorm_bitwise_when_norm_w_fused(on_16_devices):
    assert on_16_devices["rn_exact"]
    assert on_16_devices["rn_distinguishes_trunc"]


@pytest.fixture(scope="module")
def host_mesh():
    """The eight cores of one host (a fixture, not a `skipif`: asking JAX for
    devices at collection would claim the chip for the pytest parent)."""
    import jax
    from jax.sharding import Mesh

    if jax.default_backend() != "tpu" or jax.device_count() < 8:
        pytest.skip("needs the 8 TPU cores of one host")
    return Mesh(np.array(jax.devices()[:8]), ("d",))


@pytest.mark.tpu_test
@pytest.mark.multichip
@pytest.mark.parametrize("rows", ROWS)
def test_on_device_gathered_windows_are_the_replicated_projection(host_mesh, rows):
    """The Mosaic build on 8 cores (1024-column windows of the tuned 2048 x 8192
    q_b twin): every core's output bitwise fp8_skinny's on the whole weight."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

    cores, k, n = 8, 2048, 8192
    k1, k2, k3 = jax.random.split(jax.random.key(rows), 3)
    x = (jax.random.normal(k1, (rows, k)) * 2).astype(jnp.bfloat16)
    w = (jax.random.normal(k2, (k, n)) * 0.5).astype(jnp.float8_e4m3fn)
    ws = jax.random.uniform(k3, (n,), jnp.float32, 0.5, 1.5)
    ref = np.asarray(fp8_skinny.fp8_skinny_matmul(x, w, ws)).view(np.uint16)

    def body(x, w, ws, r):
        return wag.fp8_window_all_gather(
            x, w, ws, r.reshape(1), num_cores=cores, axis="d"
        )

    spec = NamedSharding(host_mesh, P("d"))
    args = [
        jax.device_put(a, spec)
        for a in (
            jnp.tile(x, (cores, 1)),
            jnp.tile(w, (cores, 1)),
            jnp.tile(ws, (cores,)),
            jnp.arange(cores, dtype=jnp.int32).reshape(cores, 1),
        )
    ]
    fn = jax.jit(
        jax.shard_map(
            body,
            mesh=host_mesh,
            in_specs=(P("d"),) * 4,
            out_specs=P("d"),
            check_vma=False,
        )
    )
    out = np.asarray(fn(*args)).reshape(cores, rows, n).view(np.uint16)
    for core in range(cores):
        np.testing.assert_array_equal(out[core], ref, err_msg=f"core {core}")


if __name__ == "__main__":
    _worker()
