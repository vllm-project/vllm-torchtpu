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
"""Every small-collective form on 8 CPU devices, in TPU interpret mode with the
race detector.

This file runs itself as a script in a subprocess, because the host device
count (XLA_FLAGS) must be set before JAX starts. It checks the DMA and
semaphore protocols, the routing of each block and the arithmetic against the
float64 oracle, not Mosaic lowering, VMEM limits or TPU timing; the same
cases run compiled on 8 TPU cores in test_all_reduce.py and
test_reduce_scatter.py.
"""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_test

N = 8
HIDDEN = N * 128
# The EP rank at each index of a TPU7x host's mesh in device-id order: local
# ranks 0..7 sit on devices [0, 1, 4, 5, 6, 7, 2, 3].
DEVICE_ID_ORDER = (0, 1, 6, 7, 2, 3, 4, 5)
RS_CASES = [
    f"{form}:{order}:{rows}"
    for form in ("hier", "flat")
    for order in ("none", "device_id")
    for rows in (3, 16)
]
AR_CASES = ["oneshot", "oneshot_xor", "hier_halves", "twoshot", "hier_2stage"]


def _worker():
    import jax
    import numpy as np
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh

    from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar
    from vllm_torchtpu.kernels.experimental.small_collectives import (
        reduce_scatter as rs,
    )

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oracle.py")
    spec = importlib.util.spec_from_file_location("small_collectives_oracle", path)
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    assert oracle.N == N
    mesh = Mesh(np.array(jax.devices()[:N]), (oracle.AXIS,))

    def interpret(body, per_core):
        from jax._src.pallas.mosaic.interpret import interpret_pallas_call

        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out = oracle.run_sharded(mesh, body, per_core, range(N))
        return out, bool(interpret_pallas_call.races.races_found)

    def exact(got, terms):
        tolerance = oracle.sum_tolerance(terms)
        return bool(oracle.within(got, oracle.exact_sum(terms), tolerance).all())

    result = {}
    for case in RS_CASES:
        form, order_name, rows = case.split(":")
        rows = int(rows)
        order = DEVICE_ID_ORDER if order_name == "device_id" else None
        per_core = oracle.blocks((N, N * rows, HIDDEN), seed=rows)

        def rs_body(x, r, form=form, order=order):
            return rs.small_reduce_scatter(
                x, num_cores=N, axis=oracle.AXIS, rank=r, form=form, block_order=order
            )

        got, races = interpret(rs_body, per_core)
        want_blocks = order or tuple(range(N))
        result[f"rs:{case}"] = {
            "exact": all(
                exact(got[c], per_core[:, b * rows : (b + 1) * rows])
                for c, b in enumerate(want_blocks)
            ),
            "races": races,
        }
    for form in AR_CASES:
        per_core = oracle.blocks((N, 8, HIDDEN), seed=7)

        def ar_body(x, r, form=form):
            return ar.small_all_reduce(
                x, num_cores=N, axis=oracle.AXIS, rank=r, form=form
            )

        got, races = interpret(ar_body, per_core)
        result[f"ar:{form}"] = {
            "exact": all(exact(got[c], per_core) for c in range(N)),
            "races": races,
        }
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        # bf16 rounded after every op in the interpreted kernel, as on the TPU.
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={N}"
        " --xla_allow_excess_precision=false",
    )
    proc = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=540,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("case", RS_CASES)
def test_reduce_scatter_on_8_devices(on_8_devices, case):
    """Core c gets the exact sum of block c, or of block block_order[c]."""
    assert on_8_devices[f"rs:{case}"] == {"exact": True, "races": False}


@pytest.mark.parametrize("form", AR_CASES)
def test_all_reduce_on_8_devices(on_8_devices, form):
    assert on_8_devices[f"ar:{form}"] == {"exact": True, "races": False}


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    # Around everything: pallas_call reads the mode when it is built.
    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
