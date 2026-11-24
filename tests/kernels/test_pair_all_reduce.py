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
"""kernels/collectives/pair_all_reduce: its topology table, and the kernel on 8
CPU devices in interpret mode with the race detector (a subprocess, since the
device count is fixed when JAX starts) against exact sums."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.collectives import pair_all_reduce

HIDDEN = 6144
# Decode buckets the kernel serves (4, 2 and 1 row blocks).
ROWS = (64, 96, 80)


# A rank-ordered mesh (DCP on): mesh index -> device id.
RANK_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)


def test_topology_table():
    t = pair_all_reduce.topology_table(range(8))
    assert t[0].tolist() == [1, 2, 4, 6, 0, 0, 0]
    assert t[7].tolist() == [6, 5, 3, 1, 1, 1, 1]


@pytest.mark.parametrize("ids", (tuple(range(8)), RANK_ORDER))
def test_topology_table_is_symmetric(ids):
    t = pair_all_reduce.topology_table(ids)
    for i in range(8):
        for col in (0, 1, 2, 3):
            assert t[t[i, col], col] == i
        # The partner is on the same chip, the other peers keep the core.
        assert ids[t[i, 0]] // 2 == ids[i] // 2
        assert all(t[t[i, col], 4] == t[i, 4] for col in (1, 2, 3))


def test_topology_table_refuses_other_shapes():
    with pytest.raises(ValueError):
        pair_all_reduce.topology_table(range(4))


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    # Mesh index i is device RANK_ORDER[i], as on a rank-ordered served mesh.
    table = pair_all_reduce.topology_table(RANK_ORDER)
    spec = P("d")

    def body(x, r):
        return pair_all_reduce.all_reduce(x, r[0], table)

    run = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )
    )
    result = {}
    for rows in ROWS:
        # Partials whose sums (and chip sums) are exact in bf16.
        x = (
            (
                np.arange(8)[:, None, None] * 5
                + np.arange(rows)[None, :, None] * 3
                + np.arange(HIDDEN)[None, None, :] // 128
            )
            % 13
            - 6
        ).astype(np.float32)
        pltpu.reset_tpu_interpret_mode_state()
        out = run(
            jax.device_put(
                jnp.asarray(x.reshape(-1, HIDDEN), jnp.bfloat16),
                NamedSharding(mesh, spec),
            ),
            jax.device_put(
                jnp.arange(8, dtype=jnp.int32).reshape(8, 1), NamedSharding(mesh, spec)
            ),
        )
        got = np.asarray(out, np.float32).reshape(8, rows, HIDDEN)
        result[str(rows)] = dict(
            exact=bool(all(np.array_equal(got[i], x.sum(0)) for i in range(8))),
            races=bool(interpret_pallas_call.races.races_found),
        )
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count=8",
    )
    proc = subprocess.run(
        [sys.executable, __file__], env=env, capture_output=True, text=True, timeout=900
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("rows", ROWS)
def test_kernel_on_8_devices(on_8_devices, rows):
    case = on_8_devices[str(rows)]
    assert case["exact"] and not case["races"], case


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
