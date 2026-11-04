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
"""small_all_reduce: the Pallas all-reduce on 8 CPU devices, and its wiring.

The device checks run this file as a script in a subprocess, because the
host device count (XLA_FLAGS) must be set before JAX starts. They use TPU
interpret mode, so they check the DMA and semaphore protocol, the slot
bookkeeping and the arithmetic of every algorithm, not Mosaic lowering or
TPU timing. Each algorithm must:

- give the exact sum on integer inputs (exact in bf16 in any order), the
  same result on every device, and an error against the float64 sum of at
  most `TOL_ULPS` bf16 ulps of sum(|x|) on normal inputs;
- stay correct over back-to-back calls in one program;
- fail the same checks for every injected fault (positive control), except
  where a fault cannot change the result (`BLIND`).
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

H = 6144
ALGOS = ("flat2", "hier1", "hier2")
FAULTS = ("rotate_peer", "drop_rank", "stale_self", "drop_tail")
# hier1 rounds twice (chip sum, then final sum), so its worst element is a
# little above one ulp of sum(|x|); XLA's all-reduce rounds at every hop.
TOL_ULPS = 1.5
# (algo, n, fault) whose fault cannot change the sum: one chip has no chip
# slot to drop, and hier2's tail tile at n = 2 is already the final sum.
BLIND = {
    ("hier1", 2, "drop_rank"),
    ("hier1", 2, "stale_self"),
    ("hier2", 2, "drop_rank"),
    ("hier2", 2, "stale_self"),
    ("hier2", 2, "drop_tail"),
}


def test_supported():
    from vllm_torchtpu.kernels.collectives import small_all_reduce as k

    assert k.supported(8, 16, H, "flat2") and k.supported(8, 16, H, "hier1")
    assert k.supported(16, 16, H, "hier2") and k.supported(16, 16, H, "flat2")
    assert not k.supported(8, 8, H, "hier1")  # below one bf16 row tile
    assert not k.supported(1, 16, H, "flat2")
    assert not k.supported(3, 16, H, "hier1")  # odd mesh: no chip pairs
    assert not k.supported(8, 16, 1000, "flat2")  # chunk not lane aligned


def test_communicator_falls_back_when_not_armed(monkeypatch):
    import torch
    from vllm.distributed.device_communicators.base_device_communicator import (
        DeviceCommunicatorBase,
    )

    from vllm_torchtpu.distributed import small_all_reduce as bridge
    from vllm_torchtpu.distributed.tpu_communicator import TpuDeviceCommunicator

    comm = object.__new__(TpuDeviceCommunicator)
    x = torch.ones(16, H, dtype=torch.bfloat16)
    monkeypatch.setattr(DeviceCommunicatorBase, "all_reduce", lambda self, t: t * 2)
    # Not armed: the usual all-reduce runs.
    assert bridge.maybe_all_reduce(comm, x) is None
    assert torch.equal(comm.all_reduce(x), x * 2)
    # Armed: the bridge's result is returned as is.
    monkeypatch.setattr(bridge, "maybe_all_reduce", lambda c, t: t * 3)
    assert torch.equal(comm.all_reduce(x), x * 3)


def test_eligibility():
    from vllm_torchtpu.distributed import small_all_reduce as bridge

    one_mib = 1 << 20
    assert bridge._eligible(8, 16, H, "hier1", one_mib)
    assert bridge._eligible(8, 64, H, "hier1", one_mib)
    assert not bridge._eligible(8, 128, H, "hier1", one_mib)  # 1.5 MiB
    assert not bridge._eligible(8, 4096, H, "hier1", one_mib)
    assert not bridge._eligible(8, 8, H, "hier1", one_mib)


def test_group_refusal():
    from vllm_torchtpu.distributed import small_all_reduce as bridge

    eight = tuple(range(8))
    assert bridge.group_refusal(eight, 8, 8) is None
    assert bridge.group_refusal((0,), 1, 8) is not None  # one rank
    assert bridge.group_refusal((0, 1, 2, 3), 8, 8) is not None  # not the world
    assert bridge.group_refusal(tuple(range(16)), 16, 8) is not None  # over the limit
    assert bridge.group_refusal(tuple(range(16)), 16, 16) is None


def _worker() -> dict:
    """Runs in the subprocess on 8 CPU devices."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.collectives import small_all_reduce as k

    devs = jax.devices()
    report = {"cells": [], "faults": []}

    def run(algo, n, t, x, fault=None, reps=1, axis=None):
        mesh = Mesh(np.asarray(devs[:n]), ("x",))
        call = k.build(k.Plan(n, t, H, algo, fault=fault, interpret=True, axis=axis))

        def body(xs, rank):
            outs, carry = [], jnp.zeros((), xs.dtype)
            for i in range(reps):
                y = (
                    call(xs[i] + carry)
                    if axis
                    else call(rank.reshape(1), xs[i] + carry)
                )
                carry = y[0, 0] * 0  # serialize the calls
                outs.append(y)
            return jnp.stack(outs)

        f = jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P("x"), P("x")),
                out_specs=P("x"),
                check_vma=False,
            )
        )
        xg = jnp.asarray(x.reshape(n * reps, t, H), jnp.bfloat16)
        rank = jnp.arange(n, dtype=jnp.int32).reshape(n, 1)
        return np.asarray(f(xg, rank).astype(jnp.float32)).reshape(n, reps, t, H)

    def inputs(n, t, kind, reps=1, seed=0):
        rng = np.random.default_rng(seed + n * 100 + t)
        if kind == "int":
            d = np.arange(n)[:, None, None, None]
            i = np.arange(reps)[None, :, None, None]
            r = np.arange(t)[None, None, :, None]
            c = np.arange(H)[None, None, None, :]
            x = ((d + 1) * 3 + r * 5 + (c // 128) * 7 + d * c + i) % 11 - 5
        else:
            x = rng.standard_normal((n, reps, t, H))
        return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32))

    def check(out, x, kind):
        want = x.astype(np.float64).sum(0)
        scale = np.abs(x.astype(np.float64)).sum(0)
        ulp = 2.0 ** (np.floor(np.log2(np.maximum(scale, 2.0**-126))) - 7)
        err = float((np.abs(out - want[None]) / ulp[None]).max())
        same = bool((out == out[:1]).all())
        ok = same and (
            np.array_equal(out, np.broadcast_to(want, out.shape))
            if kind == "int"
            else err <= TOL_ULPS
        )
        return ok, err, same

    for algo in ALGOS:
        for n, t, kind, reps in (
            (8, 16, "int", 1),
            (8, 16, "normal", 1),
            (8, 32, "normal", 1),
            (4, 16, "int", 1),
            (2, 16, "normal", 1),
            (8, 16, "normal", 3),
        ):
            x = inputs(n, t, kind, reps)
            ok, err, same = check(run(algo, n, t, x, reps=reps), x, kind)
            report["cells"].append(
                dict(
                    algo=algo, n=n, t=t, kind=kind, reps=reps, ok=ok, err=err, same=same
                )
            )
        # The production form: the rank comes from the mesh inside the kernel.
        x = inputs(8, 16, "int", 2)
        ok, err, same = check(run(algo, 8, 16, x, reps=2, axis="x"), x, "int")
        report["cells"].append(
            dict(
                algo=algo,
                n=8,
                t=16,
                kind="int",
                reps=2,
                axis="x",
                ok=ok,
                err=err,
                same=same,
            )
        )
        for n in (8, 2):
            for fault in FAULTS:
                x = inputs(n, 16, "int")
                ok, _, _ = check(run(algo, n, 16, x, fault=fault), x, "int")
                report["faults"].append(
                    dict(algo=algo, n=n, fault=fault, caught=not ok)
                )
    return report


@pytest.fixture(scope="module")
def device_report():
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=8",
    )
    proc = subprocess.run(
        [sys.executable, __file__, "--worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-4000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("REPORT ") :])


def test_sum_is_correct_on_every_algorithm(device_report):
    bad = [c for c in device_report["cells"] if not c["ok"]]
    assert not bad, bad


def test_every_fault_is_caught(device_report):
    missed = [
        f
        for f in device_report["faults"]
        if not f["caught"] and (f["algo"], f["n"], f["fault"]) not in BLIND
    ]
    assert not missed, missed


if __name__ == "__main__" and "--worker" in sys.argv:
    print("REPORT " + json.dumps(_worker()), flush=True)
