# SPDX-License-Identifier: Apache-2.0
"""Static geometry of the fused row-parallel FP8 matmul + all-reduce. The kernel itself
needs a multi-chip slice (remote DMA, barrier semaphore): it is checked on 2x2x2 against
the two production kernels, bitwise, by the t5-oproj-ar instruments. Its one-row build
is checked against the two-shot build on 16 CPU devices in TPU interpret mode (a
subprocess: the host device count must be set before JAX starts)."""

import contextlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.quantized_matmul import fp8_oproj_all_reduce as k
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny

pytestmark = pytest.mark.cpu_test
N_RANKS = 16
ONE_ROW_ROWS = (4, 16)


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


def test_owner_waves_cover_every_owner_once_after_its_last_column():
    waves = k.owner_waves(16, 6144, 2048)
    assert waves == [list(range(0, 5)), list(range(5, 10)), list(range(10, 16))]
    own = 6144 // 16
    for tile, owners in enumerate(waves):
        for j in owners:
            assert ((j + 1) * own - 1) // 2048 == tile


def test_one_row_build_needs_a_row_that_fills_a_row_tile():
    assert k.one_row_supported(6144)
    assert not k.one_row_supported(6144 + 128)


def _one_row_worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic import tpu_info
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    pltpu.get_tpu_info = tpu7x
    tpu_info.get_tpu_info = tpu7x

    def interpret(fn, *args):
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out = np.asarray(jax.block_until_ready(fn(*args)))
        return out, bool(interpret_pallas_call.races.races_found)

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    mesh = Mesh(np.asarray(devices), ("x",))
    kk, n = 1024, 6144
    rank = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)

    def build(one_row):
        def body(x, w, ws, r, nv):
            return k.fp8_matmul_all_reduce(
                x,
                w,
                ws,
                r.reshape(1),
                num_cores=N_RANKS,
                axis="x",
                waves=True,
                n_valid=nv.reshape(1) if one_row else None,
            )

        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("x"),) * 4 + (P(),),
                out_specs=P("x"),
                check_vma=False,
            )
        )

    two, one = build(False), build(True)
    result = {}
    for m in ONE_ROW_ROWS:
        k1, k2, k3 = jax.random.split(jax.random.key(m), 3)
        x = (jax.random.normal(k1, (N_RANKS * m, kk)) * 2).astype(jnp.bfloat16)
        w = (jax.random.normal(k2, (N_RANKS * kk, n)) * 0.5).astype(jnp.float8_e4m3fn)
        ws = jax.random.uniform(k3, (N_RANKS * n,), jnp.float32, 0.5, 1.5)
        case = {}
        for nv in (1, 2, m):
            args = (x, w, ws, rank, jnp.asarray([nv], jnp.int32))
            a, races_a = interpret(two, *args)
            b, races_b = interpret(one, *args)
            a = a.reshape(N_RANKS, m, n).view(np.uint16)
            b = b.reshape(N_RANKS, m, n).view(np.uint16)
            if nv == 1:
                # Row 0 bitwise; padding rows +0 (bf16 bits 0).
                same = np.array_equal(a[:, 0], b[:, 0]) and not b[:, 1:].any()
            else:
                same = np.array_equal(a, b)
            case[nv] = bool(same and not races_a and not races_b)
        result[str(m)] = case
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def one_row_on_16_devices():
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


@pytest.mark.parametrize("rows", ONE_ROW_ROWS)
def test_one_row_build_is_bitwise_the_two_shot(one_row_on_16_devices, rows):
    """n_valid 1: row 0 bitwise, padding +0; 2 and all rows: the whole output."""
    assert all(one_row_on_16_devices[str(rows)].values())


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _one_row_worker()
