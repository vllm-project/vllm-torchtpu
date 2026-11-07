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
"""kernels/collectives/tp_all_reduce: the view it picks, and the kernel on 8
CPU devices in interpret mode with the race detector (a subprocess, since the
device count is fixed when JAX starts)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.collectives import tp_all_reduce

N_RANKS = 8
# (rows, hidden): MiniMax-M3.1 decode buckets (4 and 8 share a view, 24 is
# the b3 decode), and a hidden size whose rows need padding to a lane multiple.
CASES = ((4, 6144), (8, 6144), (24, 6144), (3, 1024))


@pytest.mark.parametrize(
    "rows, hidden, want",
    [
        (8, 6144, (8, 16, 3072)),
        (4, 6144, (8, 16, 3072)),
        (24, 6144, (24, 16, 9216)),
        (3, 1024, (16, 16, 1024)),
    ],
)
def test_view_shape(rows, hidden, want):
    padded, m, w = tp_all_reduce.view_shape(rows, hidden, N_RANKS)
    assert (padded, m, w) == want
    assert m * w == padded * hidden and w % (N_RANKS * 128) == 0


def _inputs(rows, hidden):
    d = np.arange(N_RANKS)[:, None, None]
    r = np.arange(rows)[None, :, None]
    c = np.arange(hidden)[None, None, :] // 128
    return (((d + 1) * 3 + r * 5 + c * 7) % 11 - 5).astype(np.float32)


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    ranks = jnp.arange(N_RANKS, dtype=jnp.int32).reshape(N_RANKS, 1)
    run = jax.jit(
        jax.shard_map(
            lambda x, r: tp_all_reduce.all_reduce(x, r, N_RANKS),
            mesh=mesh,
            in_specs=(P("d"), P("d")),
            out_specs=P("d"),
            check_vma=False,
        )
    )
    result = {}
    for rows, hidden in CASES:
        x = _inputs(rows, hidden)
        pltpu.reset_tpu_interpret_mode_state()
        out = run(jnp.asarray(x.reshape(-1, hidden), jnp.bfloat16), ranks)
        out = np.asarray(out.astype(jnp.float32)).reshape(N_RANKS, rows, hidden)
        result[f"{rows}x{hidden}"] = dict(
            exact=all(np.array_equal(out[r], x.sum(0)) for r in range(N_RANKS)),
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
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=540
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("rows, hidden", CASES)
def test_kernel_on_8_devices(on_8_devices, rows, hidden):
    case = on_8_devices[f"{rows}x{hidden}"]
    assert case["exact"] and not case["races"], case


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
