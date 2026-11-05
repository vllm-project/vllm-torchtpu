# SPDX-License-Identifier: Apache-2.0
"""The fused row-parallel FP8 matmul + all-reduce: its static geometry, and the kernel
on 16 CPU devices in the Pallas TPU interpreter with the race detector at 16, 32 and 64
rows, bit for bit against a numpy oracle of fp8_skinny's partial summed in float32 in
rank order, and with W_UV first (uv mode) against the bf16 per-head bmm followed by the
plain kernel. The interpreter does not model VMEM, so whether 64 rows fit the scoped
limit is left to a chip."""

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as k
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

pytestmark = pytest.mark.cpu_test

N = 16
AXIS = "d"
K, N_OUT = 1024, 6144
HEADS, LATENT = 4, 512
ROWS = (16, 32, 64)
F8 = jnp.float8_e4m3fn


def test_tile_matches_fp8_skinny_o_proj_config():
    split, tile, nbuf = fp8_skinny.TUNED[(1024, 6144)]
    assert split == "n" and k.TUNED[(1024, 6144)] == (tile, nbuf)


@pytest.mark.parametrize(
    "m,kk,n,cores,ok",
    [
        (4, 1024, 6144, 16, True),
        (16, 1024, 6144, 16, True),
        (16, 1024, 6144, 8, True),
        (17, 1024, 6144, 16, False),  # above 16 rows
        (4, 2048, 16384, 16, False),  # unmeasured shape
        (4, 1024, 6144, 1, False),
        (4, 1024, 6144, 32, False),  # 192-lane chunks
    ],
)
def test_supports(m, kk, n, cores, ok):
    assert k.supports(m, kk, n, cores) is ok


@pytest.mark.parametrize(
    "m,max_m,ok",
    [
        (32, 32, True),
        (33, 32, False),
        (48, 64, True),
        (64, 64, True),
        (65, 64, False),
        (80, 96, True),
        (96, 96, True),
        (97, 96, False),
        (128, 128, False),  # above the kernel's VMEM limit
    ],
)
def test_supports_raised_row_cap(m, max_m, ok):
    assert k.supports(m, 1024, 6144, 16, max_m) is ok


def test_owner_waves_cover_every_owner_once_after_its_last_column():
    waves = k.owner_waves(16, 6144, 2048)
    assert waves == [list(range(0, 5)), list(range(5, 10)), list(range(10, 16))]
    own = 6144 // 16
    for tile, owners in enumerate(waves):
        for j in owners:
            assert ((j + 1) * own - 1) // 2048 == tile


def _partial(x, w_q, w_s):
    """fp8_skinny's bf16 partial of one rank: x [m, k] quantized per row."""
    q_max = float(jnp.finfo(F8).max)
    xf = np.asarray(x, np.float32)
    scale = np.abs(xf).max(axis=1, keepdims=True) * np.float32(1.0 / q_max)
    scale = np.where(scale == 0, np.float32(1.0), scale)
    x_q = np.clip(xf * (np.float32(1.0) / scale), -q_max, q_max).astype(F8)
    y = (x_q.astype(np.float64) @ w_q.astype(np.float64)).astype(np.float32)
    return (y * scale * w_s).astype(jnp.bfloat16)


def _all_reduce(partials):
    """The kernel's sum: float32 over the ranks in rank order, rounded once."""
    acc = np.zeros(partials[0].shape, np.float32)
    for p in partials:
        acc = acc + p.astype(np.float32)
    return acc.astype(jnp.bfloat16)


def _exact_activations(rng, m):
    """x [N, m, K] bf16 whose quantization is exact: each row 2^e times integers,
    one of them +-448 (the fp8 max) and the rest in [-16, 16], so x_q is those
    integers and the float32 matmul is exact in any order."""
    v = rng.integers(-16, 17, (N, m, K)).astype(np.float32)
    v[:, np.arange(m), rng.integers(0, K, m)] = 448.0 * rng.choice([-1, 1], m)
    return (v * 2.0 ** rng.integers(-6, 3, (N, m, 1))).astype(jnp.bfloat16)


def _run_worker():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N}"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.fixture(scope="module")
def on_16_devices():
    return _run_worker()


@pytest.mark.parametrize("m", ROWS)
def test_kernel_is_the_oracle_bit_for_bit(on_16_devices, m):
    """A chunk sent to the wrong owner or slot, a wrong quantization, rows past
    16 dropped, or a read before its DMA lands changes bits or trips the race
    detector."""
    got = on_16_devices[f"oracle_{m}"]
    assert got["equal"] and not got["races"], got


@pytest.mark.parametrize("m", ROWS)
def test_uv_mode_is_bmm_then_kernel_bit_for_bit(on_16_devices, m):
    """W_UV applied to the wrong head, latent slice or dtype, or the projected
    values read before W_UV lands, changes bits or trips the race detector."""
    got = on_16_devices[f"uv_{m}"]
    assert got["equal"] and not got["races"], got


def _worker():
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == N, devices
    mesh = Mesh(np.asarray(devices), (AXIS,))
    sharding = NamedSharding(mesh, P(AXIS))
    rng = np.random.default_rng(0)

    def run(body, *blocks):
        """body(rank [1], *this core's blocks) on every core in the interpreter:
        its outputs stacked by core, and whether the race detector fired."""

        def per_core(rank, *xs):
            out = body(rank, *(x[0] for x in xs))
            return jax.tree.map(lambda y: y[None], out)

        args = [np.arange(N, dtype=np.int32), *blocks]
        args = [jax.device_put(jnp.asarray(a), sharding) for a in args]
        fn = jax.jit(
            jax.shard_map(
                per_core,
                mesh=mesh,
                in_specs=(P(AXIS),) * len(args),
                out_specs=P(AXIS),
                check_vma=False,
            )
        )
        pltpu.reset_tpu_interpret_mode_state()
        with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
            out = jax.tree.map(np.asarray, jax.block_until_ready(fn(*args)))
        return out, bool(interpret_pallas_call.races.races_found)

    def same(a, b):
        return np.array_equal(a.view(np.uint16), np.asarray(b).view(np.uint16))

    def kernel(r, x, w, s):
        return k.fp8_matmul_all_reduce(
            x, w, s, r, num_cores=N, axis=AXIS, max_m=k.LIMIT_M
        )

    def uv_and_plain(r, x_lat, w_uv, w, s, v):
        fused = k.fp8_uv_matmul_all_reduce(
            x_lat,
            w_uv,
            w,
            s,
            r,
            num_cores=N,
            axis=AXIS,
            out_dtype=jnp.bfloat16,
            max_m=k.LIMIT_M,
        )
        return fused, kernel(r, v, w, s)

    w_q = rng.integers(-2, 3, (N, K, N_OUT)).astype(F8)
    w_s = rng.uniform(2.0**-8, 2.0**-6, (N, N_OUT)).astype(np.float32)
    w_uv = rng.integers(-3, 4, (N, HEADS, LATENT, K // HEADS)) * 2.0**-6
    w_uv = w_uv.astype(jnp.bfloat16)
    result = {}
    for m in ROWS:
        x = _exact_activations(rng, m)
        out, races = run(kernel, x, w_q, w_s)
        partials = [_partial(x[r], w_q[r], w_s[r]) for r in range(N)]
        want = _all_reduce(partials)
        result[f"oracle_{m}"] = dict(equal=all(same(o, want) for o in out), races=races)

        x_lat = rng.integers(-3, 4, (N, m, HEADS * LATENT)).astype(jnp.bfloat16)
        v = np.einsum(
            "nmhc,nhcv->nmhv",
            np.asarray(x_lat, np.float32).reshape(N, m, HEADS, LATENT),
            np.asarray(w_uv, np.float32),
        )
        v = v.reshape(N, m, K).astype(jnp.bfloat16)
        (fused, plain), races = run(uv_and_plain, x_lat, w_uv, w_q, w_s, v)
        result[f"uv_{m}"] = dict(equal=same(fused, plain), races=races)
        print(m, result[f"oracle_{m}"], result[f"uv_{m}"], file=sys.stderr, flush=True)
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__" and sys.argv[1:] == ["worker"]:
    _worker()
