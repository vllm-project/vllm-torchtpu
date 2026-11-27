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
"""kernels/collectives/pair_reduce_scatter: its topology table, and the kernel
on 8 CPU devices in interpret mode with the race detector (a subprocess, since
the device count is fixed when JAX starts) against exact sums (bf16 wire) or
the int8 wire's bound."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vllm_torchtpu.kernels.collectives import pair_reduce_scatter

HIDDEN = 384
# (rows per rank's block, chunk): several chunks per block, and one.
CASES = ((64, 32), (96, 32), (32, 32))
WIRES = pair_reduce_scatter.WIRES

# A rank-ordered mesh (DCP on): mesh index -> device id.
RANK_ORDER = (0, 1, 4, 5, 6, 7, 2, 3)


@pytest.mark.parametrize("ids", (tuple(range(8)), RANK_ORDER))
def test_topology_table(ids):
    t = pair_reduce_scatter.topology_table(ids)
    for i in range(8):
        partner, group = t[i, 0], t[i, 1:]
        assert group[0] == i
        # The partner shares the chip; the group keeps the core, one per chip.
        assert ids[partner] // 2 == ids[i] // 2 and partner != i
        assert {ids[g] % 2 for g in group} == {ids[i] % 2}
        assert sorted(ids[g] // 2 for g in group) == [0, 1, 2, 3]
        # Relations are symmetric: i is in slot j of its slot-j peer.
        for j in range(1, 4):
            assert t[group[j], 1 + j] == i


def test_topology_table_refuses_other_shapes():
    with pytest.raises(ValueError):
        pair_reduce_scatter.topology_table(range(4))


def test_chunk_rows():
    assert pair_reduce_scatter.chunk_rows(8 * 512, 6144, 8) == 64
    assert pair_reduce_scatter.chunk_rows(8 * 480, 6144, 8) == 32
    assert pair_reduce_scatter.chunk_rows(8 * 48, 6144, 8) is None
    assert pair_reduce_scatter.chunk_rows(8 * 512, 6144, 4) is None


def _worker():
    import jax
    import jax.numpy as jnp
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.asarray(sorted(jax.devices("cpu"), key=lambda d: d.id)), ("d",))
    table = pair_reduce_scatter.topology_table(RANK_ORDER)
    spec = P("d")
    result = {}
    for wire in WIRES:
        for block, chunk in CASES:
            rows = 8 * block

            def body(x, r, chunk=chunk, wire=wire):
                return pair_reduce_scatter.reduce_scatter(
                    x, r[0], table, chunk=chunk, wire=wire
                )

            run = jax.jit(
                jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(spec, spec),
                    out_specs=spec,
                    check_vma=False,
                )
            )
            # Partials whose sums (and chip sums) are exact in bf16.
            x = (
                (
                    np.arange(8)[:, None, None] * 5
                    + np.arange(rows)[None, :, None] * 3
                    + np.arange(HIDDEN)[None, None, :] // 7
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
                    jnp.arange(8, dtype=jnp.int32).reshape(8, 1),
                    NamedSharding(mesh, spec),
                ),
            )
            got = np.asarray(out, np.float32).reshape(8, block, HIDDEN)
            want = x.sum(0).reshape(8, block, HIDDEN)
            result[f"{wire}-{block}-{chunk}"] = dict(
                err=float(np.max(np.abs(got - want))),
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
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-3000:] + proc.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT ") :])


@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("case", CASES)
def test_kernel_on_8_devices(on_8_devices, wire, case):
    got = on_8_devices["{}-{}-{}".format(wire, *case)]
    # Chip sums |s| <= 12: int8 is off by up to 12 / 254 per received one,
    # and the bf16 output by up to half its spacing (1/8 below 64).
    bound = 0.0 if wire == "bf16" else 3 * 12 / 254 + 0.125
    assert got["err"] <= bound and not got["races"], got


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
