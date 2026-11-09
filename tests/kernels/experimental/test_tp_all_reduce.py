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
"""tp_all_reduce on 8 CPU devices in TPU interpret mode with the race detector.

The checks run this file as a script in a subprocess, because the host device
count (XLA_FLAGS) must be set before JAX starts. They check the DMA and
semaphore protocol and the arithmetic (exact integer sums; bf16 random inputs
bit-exact against the f32 sum in rank order rounded once), not Mosaic lowering
or TPU timing. h = 8 x 128 keeps interpret mode fast.
"""

import contextlib
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental import tp_all_reduce

N_RANKS = 8
H = N_RANKS * tp_all_reduce.LANES
ROWS = (1, 8, 13, 40)


def test_supported():
    assert tp_all_reduce.supported(8, 6144)
    assert not tp_all_reduce.supported(8, 6144 + 128)  # h / n not lane aligned
    assert not tp_all_reduce.supported(1, 6144)


def test_selftest_inputs_exact_and_column_dependent():
    x = tp_all_reduce.selftest_inputs(N_RANKS, 16, 6144)
    total = x.sum(0)
    assert np.abs(total).max() < 256  # every partial and full sum exact in bf16
    # Swapping two column chunks changes the sum.
    cw = 6144 // N_RANKS
    assert not np.array_equal(total[:, :cw], total[:, cw : 2 * cw])


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N_RANKS}",
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=540
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("rows", ROWS)
def test_exact_integer_sums(on_8_devices, rows):
    assert on_8_devices[f"int_{rows}"]


@pytest.mark.parametrize("rows", ROWS)
def test_bf16_rank_order_sum(on_8_devices, rows):
    assert on_8_devices[f"bf16_{rows}"]


def test_no_races(on_8_devices):
    assert on_8_devices["races"] is False


def _worker():
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((N_RANKS,), ("tp",))
    sharding = NamedSharding(mesh, P("tp"))
    f = jax.jit(
        jax.shard_map(
            lambda v: tp_all_reduce.all_reduce(v, N_RANKS, "tp"),
            mesh=mesh,
            in_specs=P("tp"),
            out_specs=P("tp"),
            check_vma=False,
        )
    )

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call

    result = {"races": False}

    def run(x):  # x: [n, rows, H] f32 -> [n, rows, H] f32 per rank
        n, rows, h = x.shape
        xs = jax.device_put(jnp.asarray(x.reshape(n * rows, h), jnp.bfloat16), sharding)
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out = np.asarray(jax.block_until_ready(f(xs)).astype(jnp.float32))
        result["races"] |= bool(interpret_pallas_call.races.races_found)
        return out.reshape(n, rows, h)

    rng = np.random.default_rng(0)
    for rows in ROWS:
        x = tp_all_reduce.selftest_inputs(N_RANKS, rows, H)
        got = run(x)
        result[f"int_{rows}"] = bool(
            np.array_equal(got, np.broadcast_to(x.sum(0), got.shape))
        )
        xb = np.asarray(
            jnp.asarray(rng.standard_normal((N_RANKS, rows, H)), jnp.bfloat16).astype(
                jnp.float32
            )
        )
        acc = xb[0].copy()
        for s in range(1, N_RANKS):
            acc = acc + xb[s]
        want = np.asarray(jnp.asarray(acc).astype(jnp.bfloat16).astype(jnp.float32))
        got = run(xb)
        result[f"bf16_{rows}"] = bool(
            np.array_equal(got, np.broadcast_to(want, got.shape))
        )
    print("RESULT " + json.dumps(result))


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
