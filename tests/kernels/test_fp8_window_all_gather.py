# SPDX-License-Identifier: Apache-2.0
"""The q_b column window fused with the all-gather of the windows
(TPU_MLA_LOCAL_Q_FUSED_AG) against the replicated fp8_skinny kernel, on 16 CPU
devices in TPU interpret mode with the race detector (a subprocess: the host
device count must be set before JAX starts). It covers the DMA and semaphore
protocol and the arithmetic, not Mosaic lowering or timing."""

import contextlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_window_all_gather as wag

pytestmark = pytest.mark.cpu_test
N_RANKS = 16
ROWS = (4, 16)


def test_supports_the_q_b_windows():
    assert wag.supports(4, 2048, 16384, 16)
    assert not wag.supports(17, 2048, 16384, 16)  # above the skinny rows
    assert not wag.supports(4, 2048, 16384, 32)  # 512-column windows, 1024 tiles
    assert not wag.supports(4, 1024, 6144, 16)  # o_proj tiles N by 2048


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
