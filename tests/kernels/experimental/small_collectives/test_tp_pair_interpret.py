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
"""The small all-reduce over the TP pairs of a (dp, tp) = (4, 2) mesh, and
the peer ids it addresses, on 8 CPU devices.

The checks run this file as a script in a subprocess, because the host device
count (XLA_FLAGS) must be set before JAX starts. The kernel runs in TPU
interpret mode with the race detector, so this checks the DMA and semaphore
protocol and the arithmetic, not Mosaic lowering or TPU timing. The
interpreter also accepts a peer id that names only the collective's axis, so
the multi-axis ids are checked directly.
"""

import contextlib
import io
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_test

DP, TP = 4, 2
HIDDEN = 512


def _worker():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.experimental.small_collectives import (
        all_reduce,
        common,
    )

    devices = sorted(jax.devices("cpu"), key=lambda d: d.id)
    assert len(devices) == DP * TP, devices
    mesh = Mesh(np.asarray(devices).reshape(DP, TP), ("dp_", "tp_"))
    spec = P(("dp_", "tp_"))
    tp_rank = jnp.asarray([i % TP for i in range(DP * TP)], jnp.int32)
    result = {"cases": {}}

    def run(rows, form):
        rng = np.random.default_rng(rows)
        x = jnp.asarray(rng.standard_normal((DP * TP * rows, HIDDEN)), jnp.bfloat16)

        def body(block, rank):
            with common.mesh_axes(("dp_", "tp_")):
                return all_reduce.small_all_reduce(
                    block, num_cores=TP, axis="tp_", rank=rank, form=form
                )

        f = jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
            )
        )
        pltpu.reset_tpu_interpret_mode_state()
        with contextlib.redirect_stdout(io.StringIO()):
            out = np.asarray(jax.block_until_ready(f(x, tp_rank)))
        races = bool(interpret_pallas_call.races.races_found)
        blocks = np.asarray(x.astype(jnp.float32)).reshape(DP * TP, rows, HIDDEN)
        # One rounding of the exact f32 sum of the pair's two bf16 blocks.
        want = np.stack(
            [
                np.asarray(
                    jnp.asarray(
                        blocks[TP * (i // TP)] + blocks[TP * (i // TP) + 1]
                    ).astype(jnp.bfloat16)
                )
                for i in range(DP * TP)
            ]
        )
        got = out.reshape(DP * TP, rows, HIDDEN)
        return dict(
            races_found=races,
            exact=bool(np.array_equal(got.view(np.uint16), want.view(np.uint16))),
        )

    for rows, form in ((8, "oneshot"), (16, "oneshot"), (128, "oneshot"), (8, None)):
        result["cases"][f"{rows}:{form}"] = run(rows, form)

    def ids_body(dummy):
        with common.mesh_axes(("dp_", "tp_")):
            by_name = common.peer(1, "tp_", True)
            in_order = common.peer(1, "tp_", False)
        outside = common.peer(1, "tp_", True)
        row = jnp.stack(
            [
                jax.lax.axis_index("dp_"),
                by_name["dp_"],
                by_name["tp_"],
                in_order[0],
                in_order[1],
                jnp.int32(len(in_order)),
                jnp.int32(len(outside)),
            ]
        ).astype(jnp.int32)
        return row.reshape(1, -1) + dummy[:, :1] * 0

    ids = np.asarray(
        jax.jit(
            jax.shard_map(
                ids_body, mesh=mesh, in_specs=(spec,), out_specs=spec, check_vma=False
            )
        )(jnp.zeros((DP * TP, 1), jnp.int32))
    )
    result["peer_ids"] = ids.tolist()
    print("RESULT " + json.dumps(result))


@pytest.fixture(scope="module")
def on_8_devices():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{flags} --xla_force_host_platform_device_count={DP * TP}",
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


@pytest.mark.parametrize("case", ["8:oneshot", "16:oneshot", "128:oneshot", "8:None"])
def test_tp_pair_all_reduce_is_the_pair_sum(on_8_devices, case):
    got = on_8_devices["cases"][case]
    assert not got["races_found"], got
    assert got["exact"], got


def test_peer_ids_name_every_mesh_axis_inside_mesh_axes(on_8_devices):
    for row in on_8_devices["peer_ids"]:
        dp_index, by_name_dp, by_name_tp, first, second, n_order, n_outside = row
        # By name: the peer's TP index and this core's own DP index.
        assert (by_name_dp, by_name_tp) == (dp_index, 1)
        # In mesh order: (dp, tp).
        assert (first, second, n_order) == (dp_index, 1, 2)
        # Outside mesh_axes, the one-axis id as before.
        assert n_outside == 1


if __name__ == "__main__":
    from jax.experimental.pallas import tpu as pltpu

    with pltpu.force_tpu_interpret_mode(pltpu.InterpretParams(detect_races=True)):
        _worker()
