# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""moe_tp_decode's in-kernel all-reduce: the one-row path (one round,
TPU_MOE_TP_DECODE_AR_ONE_SHOT) against the two-shot paths on 16 CPU devices.

The check runs this file as a script in a subprocess (the host device count
must be set before JAX starts), in TPU interpret mode with the race detector:
it covers the DMA and semaphore protocol and the arithmetic, not Mosaic
lowering or timing.
"""

import contextlib
import io
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode import all_reduce

pytestmark = pytest.mark.cpu_test
N_RANKS = 16
# (rows, hidden): the GLM-5.2 decode bucket, and a 32-row bucket at another width.
SHAPES = ((16, 6144), (32, 4096))


def _tpu7x():
    return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)


def _interpret(fn, *args):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    pltpu.reset_tpu_interpret_mode_state()
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(jax.block_until_ready(fn(*args)))
    return out, bool(interpret_pallas_call.races.races_found)


def _worker():
    from jax._src.pallas.mosaic import tpu_info
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    pltpu.get_tpu_info = _tpu7x
    tpu_info.get_tpu_info = _tpu7x
    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    mesh = Mesh(np.asarray(devices), ("d",))
    rank_ids = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)

    def runner(spec):
        call = all_reduce.standalone(spec)

        def body(x, rank, n_valid):
            return call(jnp.stack([n_valid.reshape(()), rank.reshape(())]), x)

        run = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("d"), P("d"), P()),
                out_specs=P("d"),
                check_vma=False,
            )
        )

        def go(x, n_valid):
            out, races = _interpret(
                run,
                jnp.asarray(x.reshape(N_RANKS * spec.m, spec.h), jnp.bfloat16),
                rank_ids,
                jnp.asarray([n_valid], jnp.int32),
            )
            return out.reshape(N_RANKS, spec.m, spec.h), races

        return go

    result = {}
    for m, h in SHAPES:
        two = all_reduce.make_spec(N_RANKS, m, h)
        one = all_reduce.make_spec(N_RANKS, m, h, one_shot=True)
        exact = all_reduce.selftest_inputs(one)
        selftest = True
        for n_valid in all_reduce.selftest_row_counts(one):
            out, races = runner(one)(exact, n_valid)
            selftest &= not races and all(
                all_reduce.selftest_ok(one, exact, out[r], n_valid)
                for r in range(N_RANKS)
            )
        # Partials whose f32 sums round: any other summation order shows.
        noisy = np.asarray(
            jax.random.normal(jax.random.key(m), (N_RANKS, m, h)) * 3, jnp.bfloat16
        )
        bitwise = {}
        for n_valid in (1, 2, m):
            a, races_a = runner(two)(noisy, n_valid)
            b, races_b = runner(one)(noisy, n_valid)
            bitwise[n_valid] = bool(
                not races_a
                and not races_b
                and np.array_equal(a.view(np.uint16), b.view(np.uint16))
            )
        result[f"{m}x{h}"] = dict(selftest=bool(selftest), bitwise=bitwise)
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


def test_row_counts_put_the_one_row_path_first(monkeypatch):
    monkeypatch.setattr(pltpu, "get_tpu_info", _tpu7x)
    spec = all_reduce.make_spec(N_RANKS, 16, 6144, one_shot=True)
    assert all_reduce.selftest_row_counts(spec) == [16, 1, 5, 0]
    assert all_reduce.kernel_name(spec).endswith("-os")


@pytest.mark.parametrize("m, h", SHAPES)
def test_one_row_path_passes_the_selftest(on_16_devices, m, h):
    assert on_16_devices[f"{m}x{h}"]["selftest"]


@pytest.mark.parametrize("m, h", SHAPES)
def test_one_row_path_is_bitwise_the_two_shot(on_16_devices, m, h):
    """n_valid 1 takes the one-row path; 2 and m keep the two-shot's paths."""
    assert all(on_16_devices[f"{m}x{h}"]["bitwise"].values())


if __name__ == "__main__":
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
